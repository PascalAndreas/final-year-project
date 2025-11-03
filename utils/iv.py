"""
Implied Volatility Surface Construction and Visualization

Fetches futures and options orderbook data, computes implied volatilities,
and creates interactive 3D surface plots with time slider.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import plotly.graph_objects as go
from scipy.interpolate import RBFInterpolator
import warnings
import time

from okx import fetch_market_data, bin_orderbook
from .helpers import parse_option_name, parse_future_name
from .black import black76_implied_volatility, black76_implied_volatility_vectorized
from .forward import compute_segment_carry, interpolate_forward

def _build_forward_interpolator(swap_df: pd.DataFrame, futures_df: pd.DataFrame):
    """Build forward interpolator for all timestamps. Returns dict: timestamp -> segments_df."""
    if swap_df.empty or futures_df.empty:
        return {}
    
    futures_df['expiry'] = futures_df['symbol'].apply(lambda x: parse_future_name(x)[1])
    interpolators = {}
    
    for _, swap_row in swap_df.iterrows():
        time_bin = swap_row['time_bin']
        swap_bid, swap_ask = swap_row['bid_1_px'], swap_row['ask_1_px']
        futures_at_time = futures_df[futures_df['time_bin'] == time_bin].copy()
        
        if len(futures_at_time) > 0:
            segments = compute_segment_carry(futures_at_time, swap_bid, swap_ask, time_bin)
            interpolators[swap_row['timestamp']] = (segments, swap_bid, swap_ask, time_bin)
    
    return interpolators


def _compute_option_iv(options_df: pd.DataFrame, interpolators: dict, 
                       price_column: str, forward_bound: str,
                       risk_free_rate: float = 0.0, verbose: bool = False, 
                       diagnostics: bool = False) -> pd.DataFrame:
    """Compute IV using interpolated forwards. forward_bound: 'lower' for bid, 'upper' for ask."""
    if options_df.empty or not interpolators:
        return pd.DataFrame()
    
    options_df[['inst_family', 'expiry', 'strike', 'option_type']] = pd.DataFrame(
        options_df['symbol'].apply(parse_option_name).tolist(), index=options_df.index
    )
    
    forward_data = []
    for _, opt in options_df.iterrows():
        if opt['timestamp'] not in interpolators:
            continue
        segments, swap_bid, swap_ask, swap_time = interpolators[opt['timestamp']]
        T = (opt['expiry'] - swap_time).total_seconds() / (365.25 * 24 * 3600)
        
        F_bounds = interpolate_forward(T, segments)
        if F_bounds is None:
            continue
        
        forward_price = F_bounds[0] if forward_bound == 'lower' else F_bounds[1]
        forward_data.append({'timestamp': opt['timestamp'], 'expiry': opt['expiry'], 
                           'strike': opt['strike'], 'option_type': opt['option_type'],
                           'symbol': opt['symbol'], price_column: opt[price_column],
                           'forward_price': forward_price})
    
    if not forward_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(forward_data)
    df['tenor_days'] = ((df['expiry'] - pd.to_datetime(df['timestamp'], unit='ms')).dt.total_seconds() / (24 * 3600))
    df['log_moneyness'] = np.log(df['strike'] / df['forward_price'])
    
    valid = (df['tenor_days'] > 0) & (df['strike'] > 0) & (df['forward_price'] > 0)
    df = df[valid].copy()
    
    df['market_price_usd'] = df[price_column] * df['forward_price']
    df['implied_vol'] = black76_implied_volatility_vectorized(
        market_price=df['market_price_usd'].values, F=df['forward_price'].values,
        K=df['strike'].values, T=(df['tenor_days'] / 365.25).values,
        r=risk_free_rate, option_type=df['option_type'].values, verbose=diagnostics
    )
    
    failed_count = df['implied_vol'].isna().sum()
    if verbose and failed_count > 0:
        print(f"    Failed IV calculations: {failed_count}/{len(df)}")
    
    df = df.dropna(subset=['implied_vol'])
    return df[['timestamp', 'symbol', 'expiry', 'strike', 'option_type',
               price_column, 'forward_price', 'tenor_days', 'log_moneyness', 'implied_vol']]


def construct_iv_surface(
    inst_family: str,
    start_date: datetime,
    num_days: int = 1,
    time_step_minutes: int = 5,
    risk_free_rate: float = 0.0,
    verbose: bool = True,
    diagnostics: bool = False,
    use_bid_ask: bool = False
) -> pd.DataFrame:
    """Construct IV surface using interpolated forwards. Computes IV for ALL options, not just those with listed futures."""
    all_iv_data = []
    dates_to_fetch = [start_date + timedelta(days=i) for i in range(num_days)]
    
    for date in tqdm(dates_to_fetch, desc="Processing dates", disable=not verbose):
        day_end = date + timedelta(days=1) - timedelta(seconds=1)
        
        # Fetch SWAP data for forward curve anchor
        swap_df = fetch_market_data(
            '6', 'SWAP', inst_family, date, day_end, 'daily',
            verbose=verbose, depth=1,
            include_criterion=lambda filename: filename.startswith(f'{inst_family}-SWAP'),
            process_fn=lambda df: bin_orderbook(df, f'{time_step_minutes}min')
        )
        
        # Fetch futures for forward curve construction
        futures_df = fetch_market_data(
            '6', 'FUTURES', inst_family, date, day_end, 'daily',
            verbose=verbose, depth=1,
            include_criterion=lambda filename: filename.startswith(f'{inst_family}-'),
            process_fn=lambda df: bin_orderbook(df, f'{time_step_minutes}min')
        )
        
        # Build forward interpolators
        interpolators = _build_forward_interpolator(swap_df, futures_df)
        if not interpolators:
            if verbose:
                print(f"  {date.date()}: No forward curve data, skipping")
            continue
        
        if verbose:
            print(f"  {date.date()}: Built {len(interpolators)} forward curves")
        
        # Fetch ALL options (no expiry filter)
        options_df = fetch_market_data(
            '6', 'OPTION', inst_family, date, day_end, 'daily',
            verbose=verbose, depth=1 if use_bid_ask else 0,
            process_fn=lambda df: bin_orderbook(df, f'{time_step_minutes}min')
        )
        
        if options_df.empty:
            if verbose:
                print(f"  {date.date()}: No options data, skipping")
            continue
        
        # Compute IVs using interpolated forwards
        if use_bid_ask:
            bid_data = _compute_option_iv(options_df, interpolators, 'bid_1_px', 'lower', 
                                         risk_free_rate, verbose, diagnostics).rename(columns={'implied_vol': 'bid_iv'})
            ask_data = _compute_option_iv(options_df, interpolators, 'ask_1_px', 'upper',
                                         risk_free_rate, verbose, diagnostics).rename(columns={'implied_vol': 'ask_iv'})
            
            merge_cols = ['timestamp', 'symbol', 'expiry', 'strike', 'option_type',
                         'forward_price', 'tenor_days', 'log_moneyness']
            iv_data = bid_data.merge(ask_data[merge_cols + ['ask_1_px', 'ask_iv']], on=merge_cols, how='outer')
        else:
            # For mid, use average of bounds
            iv_data = _compute_option_iv(options_df, interpolators, 'mid_price', 'lower',
                                        risk_free_rate, verbose, diagnostics)
        
        all_iv_data.append(iv_data)
        if verbose:
            print(f"  {date.date()}: Computed {len(iv_data)} IVs")
    
    if not all_iv_data:
        if verbose:
            print("No IV data computed")
        return pd.DataFrame()
    
    result = pd.concat(all_iv_data, ignore_index=True)
    if verbose:
        print(f"\n✓ Total: {len(result)} IV points across {len(dates_to_fetch)} days")
    
    return result


def create_iv_surface_plot(
    iv_df: pd.DataFrame,
    grid_resolution: int = 40,
    title: str = "Implied Volatility Surface",
    smoothing: float = 0.0,
    kernel: str = 'thin_plate_spline'
) -> go.Figure:
    """Create 3D IV surface with time slider using RBF interpolation.
    
    Args:
        smoothing: Smoothing parameter for RBF (0=interpolation, >0=regularization)
        kernel: RBF kernel - 'thin_plate_spline' (default, smooth), 'cubic', 'quintic', 'multiquadric', 'linear'
    """
    
    # Detect mode and prep data
    has_bid_ask = 'bid_iv' in iv_df.columns and 'ask_iv' in iv_df.columns
    iv_df = iv_df.copy()
    iv_df['datetime'] = pd.to_datetime(iv_df['timestamp'], unit='ms')
    
    # Compute axis ranges once
    x_min, x_max = iv_df['log_moneyness'].quantile([0.01, 0.99])
    y_min, y_max = iv_df['tenor_days'].quantile([0.01, 0.99])
    
    if has_bid_ask:
        all_ivs = pd.concat([iv_df['bid_iv'].dropna(), iv_df['ask_iv'].dropna()])
        z_min, z_max = all_ivs.quantile([0.01, 0.99])
    else:
        z_min, z_max = iv_df['implied_vol'].quantile([0.01, 0.99])
    
    # Add padding
    x_min, x_max = x_min - (x_max - x_min) * 0.05, x_max + (x_max - x_min) * 0.05
    y_min, y_max = max(0.1, y_min * 0.9), y_max * 1.1
    z_min, z_max = max(0, z_min - (z_max - z_min) * 0.05), z_max + (z_max - z_min) * 0.05
    
    # Sequential frame generation
    unique_times = sorted(iv_df['datetime'].unique())
    frames = []
    
    surfaces_created = 0
    for time_val in tqdm(unique_times, desc="Processing frames"):
        bin_data = iv_df[iv_df['datetime'] == time_val]
        frame_objects = []
        
        # Surface configs: (surf_type, iv_col, colorscale, showscale, scatter_color)
        if has_bid_ask:
            configs = [
                ('bid', 'bid_iv', 'Blues', False, 'blue'),
                ('ask', 'ask_iv', 'Reds', True, 'red')
            ]
        else:
            configs = [('mid', 'implied_vol', 'Viridis', True, 'green')]
        
        for surf_type, iv_col, colorscale, showscale, scatter_color in configs:
            # Get data with valid IVs
            data = bin_data[bin_data[iv_col].notna()].copy()
            if len(data) < 4:  # Need at least 4 points for RBF
                continue
            
            # Fit RBF surface (debug first frame only)
            debug = (surfaces_created == 0)
            log_m_grid, tenor_grid, iv_grid = _fit_rbf_surface(
                data, iv_col, grid_resolution, smoothing, kernel, debug
            )
            
            if debug and log_m_grid is not None:
                print(f"  Output IV surface: range=[{np.min(iv_grid):.4f}, {np.max(iv_grid):.4f}]")
            
            if log_m_grid is not None:
                surfaces_created += 1
                
                # Create surface and scatter
                surface = go.Surface(
                    x=log_m_grid, y=tenor_grid, z=iv_grid,
                    colorscale=colorscale, cmin=z_min, cmax=z_max,
                    colorbar=dict(title="IV", x=1.02) if showscale else None,
                    showscale=showscale, name=f'{surf_type.title()} IV', opacity=0.8
                )
                scatter = go.Scatter3d(
                    x=data['log_moneyness'], y=data['tenor_days'], z=data[iv_col],
                    mode='markers', marker=dict(size=2, color=scatter_color),
                    name=f'{surf_type.title()} Points'
                )
                frame_objects.extend([surface, scatter])
        
        if frame_objects:
            frame_tenors = sorted(bin_data['tenor_days'].unique())
            frames.append(go.Frame(
                data=frame_objects, name=str(time_val),
                layout=dict(scene_yaxis_tickvals=frame_tenors,
                           scene_yaxis_ticktext=[f"{t:.2f}" for t in frame_tenors])
            ))
    
    if not frames:
        print("No valid frames generated")
        return None
    
    print(f"\n✓ Generated {len(frames)} frames with {surfaces_created} total surfaces")
    print(f"First frame has {len(frames[0].data)} traces:")
    for trace in frames[0].data:
        print(f"  - {trace.name}: {type(trace).__name__}")
        if hasattr(trace, 'z') and trace.z is not None:
            z_data = trace.z if isinstance(trace.z, np.ndarray) else np.array(trace.z)
            if len(z_data.shape) == 2:
                print(f"    Shape: {z_data.shape}, Range: [{np.min(z_data):.4f}, {np.max(z_data):.4f}]")
    
    # Build figure
    first_tenors = sorted(iv_df[iv_df['datetime'] == unique_times[0]]['tenor_days'].unique())
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Log Moneyness ln(K/F)',
            yaxis_title='Tenor (days)',
            zaxis_title='Implied Volatility',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(type='log', range=[np.log10(y_min), np.log10(y_max)], tickmode='array',
                      tickvals=first_tenors, ticktext=[f"{t:.2f}" for t in first_tenors]),
            zaxis=dict(range=[z_min, z_max]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        updatemenus=[{
            'type': 'buttons', 'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
            ], 'x': 0.1, 'y': 0
        }],
        sliders=[{
            'active': 0, 'yanchor': 'top', 'y': 0, 'xanchor': 'left', 'x': 0.3,
            'currentvalue': {'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
            'steps': [{'args': [[f.name], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}],
                      'method': 'animate', 'label': str(f.name)[:16]} for f in frames]
        }],
        height=700, width=1000
    )
    return fig


def _fit_rbf_surface(df: pd.DataFrame, iv_col: str, grid_resolution: int = 40, 
                     smoothing: float = 0.0, kernel: str = 'thin_plate_spline', debug: bool = False):
    """Fit RBF interpolator to IV data and construct smooth surface. Returns (grid_x, grid_y, grid_z)."""
    
    # Clean data
    df_clean = df[
        np.isfinite(df['log_moneyness']) & 
        np.isfinite(df['tenor_days']) & 
        np.isfinite(df[iv_col]) &
        (df['tenor_days'] > 0) &
        (df[iv_col] > 0)
    ].copy()
    
    if debug and len(df_clean) > 0:
        print(f"\nDEBUG _fit_rbf_surface:")
        print(f"  Input IVs: count={len(df_clean)}, range=[{df_clean[iv_col].min():.4f}, {df_clean[iv_col].max():.4f}]")
        print(f"  Log moneyness range: [{df_clean['log_moneyness'].min():.4f}, {df_clean['log_moneyness'].max():.4f}]")
        print(f"  Tenor range: [{df_clean['tenor_days'].min():.2f}, {df_clean['tenor_days'].max():.2f}] days")
        print(f"  Kernel: {kernel}, Smoothing: {smoothing}")
    
    if len(df_clean) < 4:
        return None, None, None
    
    # Prepare input points (log_moneyness, log(tenor_days)) and values
    # Use log(tenor) for better RBF behavior across wide tenor ranges
    X = np.column_stack([
        df_clean['log_moneyness'].values,
        np.log(df_clean['tenor_days'].values)
    ])
    y = df_clean[iv_col].values
    
    # Create 2D grid for evaluation
    log_m_min, log_m_max = df_clean['log_moneyness'].quantile([0.05, 0.95])
    log_m_range = log_m_max - log_m_min
    log_m_min -= log_m_range * 0.1
    log_m_max += log_m_range * 0.1
    log_m_grid = np.linspace(log_m_min, log_m_max, grid_resolution)
    
    # For tenor, use log-spaced grid in original tenor space
    tenor_min, tenor_max = df_clean['tenor_days'].quantile([0.05, 0.95])
    tenor_min = max(0.1, tenor_min * 0.9)
    tenor_max = tenor_max * 1.1
    tenor_grid_1d = np.logspace(np.log10(tenor_min), np.log10(tenor_max), grid_resolution)
    
    # Create meshgrid
    log_m_mesh, tenor_mesh = np.meshgrid(log_m_grid, tenor_grid_1d)
    
    # Prepare evaluation points (using log(tenor))
    X_eval = np.column_stack([
        log_m_mesh.ravel(),
        np.log(tenor_mesh.ravel())
    ])
    
    try:
        # Fit RBF interpolator
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            rbf = RBFInterpolator(X, y, kernel=kernel, smoothing=smoothing)
            
            # Evaluate on grid
            iv_grid_flat = rbf(X_eval)
            iv_grid = iv_grid_flat.reshape(log_m_mesh.shape)
            
            # Clamp values to reasonable range (avoid extrapolation artifacts)
            iv_min, iv_max = np.percentile(y, [1, 99])
            iv_buffer = (iv_max - iv_min) * 0.3
            iv_grid = np.clip(iv_grid, max(0, iv_min - iv_buffer), iv_max + iv_buffer)
            
            if debug:
                print(f"  RBF fitted successfully")
                print(f"  Evaluation grid: {grid_resolution}x{grid_resolution} = {grid_resolution**2} points")
                print(f"  Output IV range (clamped): [{np.min(iv_grid):.4f}, {np.max(iv_grid):.4f}]")
            
            return log_m_mesh, tenor_mesh, iv_grid
            
    except Exception as e:
        if debug:
            print(f"  RBF fitting failed: {e}")
        return None, None, None