"""
Implied Volatility Surface Construction and Visualization

Fetches futures and options orderbook data, computes implied volatilities,
and creates interactive 3D surface plots with time slider.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from tqdm.auto import tqdm
import plotly.graph_objects as go

from .api import fetch_market_data
from .black import black76_implied_volatility, black76_implied_volatility_vectorized
from .helpers import parse_option_name, parse_future_name, bin_orderbook

def _process_futures(futures_df: pd.DataFrame) -> pd.DataFrame:
    """Process binned futures to get forward prices by expiry and time.
    Assumes orderbook is already binned and mid_price is calculated."""
    if futures_df.empty:
        return pd.DataFrame(columns=['timestamp', 'expiry', 'forward_price'])
    
    # Parse expiries and use mid_price as forward price
    futures_df['expiry'] = futures_df['symbol'].apply(lambda x: parse_future_name(x)[1])
    
    return pd.DataFrame({
        'timestamp': futures_df['timestamp'],
        'expiry': futures_df['expiry'],
        'forward_price': futures_df['mid_price']
    })


def _compute_option_iv(options_df: pd.DataFrame, forward_prices: pd.DataFrame, 
                       risk_free_rate: float = 0.0, verbose: bool = False, 
                       diagnostics: bool = False) -> pd.DataFrame:
    """Compute implied volatility for binned options using forward prices.
    Assumes orderbook is already binned and mid_price is calculated."""
    if options_df.empty or forward_prices.empty:
        return pd.DataFrame()
    
    # Parse option instruments (mid_price already calculated in trim_orderbook)
    options_df[['inst_family', 'expiry', 'strike', 'option_type']] = pd.DataFrame(
        options_df['symbol'].apply(parse_option_name).tolist(), index=options_df.index
    )
    
    # Merge with forward prices
    df = options_df.merge(
        forward_prices, 
        on=['timestamp', 'expiry'],
        how='inner'
    )
    
    # Vectorized calculations
    df['tenor_days'] = (
        (df['expiry'] - pd.to_datetime(df['timestamp'], unit='ms')).dt.total_seconds() 
        / (24 * 3600)
    )
    df['log_moneyness'] = np.log(df['strike'] / df['forward_price'])
    
    # Filter invalid entries
    valid = (df['tenor_days'] > 0) & (df['strike'] > 0) & (df['forward_price'] > 0)
    df = df[valid].copy()
    
    # Convert option price from BTC to USD by multiplying by forward price
    df['market_price_usd'] = df['mid_price'] * df['forward_price']
    
    # Vectorized IV calculation
    df['implied_vol'] = black76_implied_volatility_vectorized(
        market_price=df['market_price_usd'].values,
        F=df['forward_price'].values,
        K=df['strike'].values,
        T=(df['tenor_days'] / 365.25).values,  # Convert days to years for Black-76
        r=risk_free_rate,
        option_type=df['option_type'].values,
        verbose=diagnostics  # Pass diagnostics flag to solver
    )
    
    # Report failed IV calculations
    failed_count = df['implied_vol'].isna().sum()
    if verbose and failed_count > 0:
        print(f"    Failed IV calculations: {failed_count}/{len(df)}")
    
    df = df.dropna(subset=['implied_vol'])
    
    return df[[
        'timestamp', 'symbol', 'expiry', 'strike', 'option_type',
        'mid_price', 'forward_price', 'tenor_days', 'log_moneyness', 'implied_vol'
    ]]


def construct_iv_surface(
    inst_family: str,
    start_date: datetime,
    num_days: int = 1,
    time_step_minutes: int = 5,
    risk_free_rate: float = 0.0,
    verbose: bool = True,
    diagnostics: bool = False
) -> pd.DataFrame:
    """Construct IV surface by fetching futures and options day-by-day.
    Returns DataFrame with timestamp, expiry, strike, option_type, implied_vol, etc."""
    all_iv_data = []
    
    # Create list of dates to fetch
    dates_to_fetch = [start_date + timedelta(days=i) for i in range(num_days)]
    
    for date in tqdm(dates_to_fetch, desc="Processing dates", disable=not verbose):
        day_end = date + timedelta(days=1) - timedelta(seconds=1)
        
        # Step 1: Fetch futures orderbook (depth=0 to get mid_price only)
        futures_df = fetch_market_data(
            '6', 'FUTURES', inst_family, date, day_end, 'daily', 
            verbose=verbose,
            depth=0,
            process_fn=lambda df: bin_orderbook(df, f'{time_step_minutes}min')
        )
        
        # Step 2: Process futures to get forward prices and available expiries
        forward_prices = _process_futures(futures_df)
        available_expiries = set(forward_prices['expiry'].unique())
        if verbose:
            print(f"  {date.date()}: {len(available_expiries)} available expiries")
        
        # Step 3: Fetch options, filtering for expiries with futures
        def filter_by_expiry(filename: str) -> bool:
            """Filter options by expiries that have corresponding futures."""
            try:
                _, expiry, _, _ = parse_option_name(filename)
                return expiry in available_expiries
            except:
                return False
        
        options_df = fetch_market_data(
            '6', 'OPTION', inst_family, date, day_end, 'daily', 
            verbose=verbose,
            depth=0,
            include_criterion=filter_by_expiry,
            process_fn=lambda df: bin_orderbook(df, f'{time_step_minutes}min')
        )
        
        if options_df.empty:
            if verbose:
                print(f"  {date.date()}: No options data, skipping")
            continue
        
        # Step 4: Compute IVs
        iv_data = _compute_option_iv(options_df, forward_prices, risk_free_rate, verbose, diagnostics)
        
        if not iv_data.empty:
            all_iv_data.append(iv_data)
            if verbose:
                print(f"  {date.date()}: Computed {len(iv_data)} IVs")
    
    if not all_iv_data:
        if verbose:
            print("No IV data computed")
        return pd.DataFrame()
    
    # Combine all data
    result = pd.concat(all_iv_data, ignore_index=True)
    
    if verbose:
        print(f"\n✓ Total: {len(result)} IV points across {len(dates_to_fetch)} days")
    
    return result


def create_iv_surface_plot(
    iv_df: pd.DataFrame,
    grid_resolution: int = 40,
    title: str = "Implied Volatility Surface"
) -> go.Figure:
    """Create interactive 3D IV surface plot with time slider.
    Data should already be time-binned from construct_iv_surface."""
    if iv_df.empty:
        print("No data to plot")
        return go.Figure()
    
    # Convert timestamps and filter valid IVs
    iv_df = iv_df.copy()
    iv_df['datetime'] = pd.to_datetime(iv_df['timestamp'], unit='ms')
    iv_df = iv_df[np.isfinite(iv_df['implied_vol']) & (iv_df['implied_vol'] > 0)]
    
    if len(iv_df) == 0:
        print("No valid IV data to plot")
        return go.Figure()
    
    # Get unique timestamps (data is already binned by construct_iv_surface)
    unique_times = sorted(iv_df['datetime'].unique())
    
    # Compute fixed axis ranges across all data (prevents jumpy axes)
    x_min, x_max = iv_df['log_moneyness'].quantile([0.01, 0.99])
    y_min, y_max = iv_df['tenor_days'].quantile([0.01, 0.99])
    z_min, z_max = iv_df['implied_vol'].quantile([0.01, 0.99])
    
    # Add padding to ranges
    x_range = x_max - x_min
    z_range = z_max - z_min
    
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    # For log scale, use multiplicative padding
    y_min = max(0.1, y_min * 0.9)  # Ensure positive for log scale
    y_max = y_max * 1.1
    z_min = max(0, z_min - z_range * 0.05)
    z_max += z_range * 0.05
    
    # Create frames for each unique timestamp
    frames = []
    for time_val in tqdm(unique_times, desc="Creating frames"):
        bin_data = iv_df[iv_df['datetime'] == time_val]
        
        if len(bin_data) < 3:
            frames.append(go.Frame(
                data=[
                    go.Surface(x=[], y=[], z=[], showscale=False, name='IV Surface'),
                    go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Data Points')
                ],
                name=str(time_val),
                layout=dict(
                    scene_yaxis_tickvals=[],
                    scene_yaxis_ticktext=[]
                )
            ))
            continue
        
        # Interpolate onto grid
        log_m_grid, tenor_grid, iv_grid = _interpolate_surface(bin_data, grid_resolution)
        
        if log_m_grid is None:
            frames.append(go.Frame(
                data=[
                    go.Surface(x=[], y=[], z=[], showscale=False, name='IV Surface'),
                    go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Data Points')
                ],
                name=str(time_val),
                layout=dict(
                    scene_yaxis_tickvals=[],
                    scene_yaxis_ticktext=[]
                )
            ))
            continue
        
        # Create surface
        surface = go.Surface(
            x=log_m_grid,
            y=tenor_grid,  # Already in days
            z=iv_grid,
            colorscale='Viridis',
            cmin=z_min,
            cmax=z_max,
            colorbar=dict(title="IV", x=1.02),
            showscale=True,
            name='IV Surface'
        )
        
        # Create scatter points for actual data
        scatter = go.Scatter3d(
            x=bin_data['log_moneyness'],
            y=bin_data['tenor_days'],
            z=bin_data['implied_vol'],
            mode='markers',
            marker=dict(
                size=3,
                color='red',
                symbol='circle'
            ),
            name='Data Points'
        )
        
        # Get unique tenors for this frame only
        frame_tenors = sorted(bin_data['tenor_days'].unique())
        frame_tenor_labels = [f"{t:.2f}" for t in frame_tenors]
        
        frames.append(go.Frame(
            data=[surface, scatter], 
            name=str(time_val),
            layout=dict(
                scene_yaxis_tickvals=frame_tenors,
                scene_yaxis_ticktext=frame_tenor_labels
            )
        ))
    
    # Create figure with initial frame data and layout
    initial_layout = frames[0].layout if frames and hasattr(frames[0], 'layout') else None
    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)
    
    # Get initial tenor ticks from first frame
    initial_tenors = []
    initial_tenor_labels = []
    if frames and len(frames[0].data) > 0:
        first_bin_data = iv_df[iv_df['datetime'] == unique_times[0]]
        if not first_bin_data.empty:
            initial_tenors = sorted(first_bin_data['tenor_days'].unique())
            initial_tenor_labels = [f"{t:.2f}" for t in initial_tenors]
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Log Moneyness ln(K/F)',
            yaxis_title='Tenor (days)',
            zaxis_title='Implied Volatility',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(
                type='log',  # Logarithmic scale
                range=[np.log10(y_min), np.log10(y_max)],
                tickmode='array',
                tickvals=initial_tenors,
                ticktext=initial_tenor_labels
            ),
            zaxis=dict(range=[z_min, z_max]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                 'fromcurrent': True, 'transition': {'duration': 300}}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                   'mode': 'immediate', 'transition': {'duration': 0}}]}
            ],
            'x': 0.1, 'y': 0
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top', 'y': 0,
            'xanchor': 'left', 'x': 0.3,
            'currentvalue': {'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
            'steps': [
                {'args': [[frame.name], {'frame': {'duration': 300, 'redraw': True},
                                         'mode': 'immediate', 'transition': {'duration': 300}}],
                 'method': 'animate', 'label': str(frame.name)[:16]}
                for frame in frames
            ]
        }],
        height=700,
        width=1000
    )
    
    return fig


def _interpolate_surface(df: pd.DataFrame, grid_resolution: int = 40):
    """Interpolate sparse IV data onto regular grid.
    Returns log_moneyness_grid, tenor_grid, iv_grid as 2D arrays."""
    if len(df) < 3:
        return None, None, None
    
    # Remove NaN/inf values
    df_clean = df[
        np.isfinite(df['log_moneyness']) & 
        np.isfinite(df['tenor_days']) & 
        np.isfinite(df['implied_vol']) &
        (df['tenor_days'] > 0)
    ].copy()
    
    if len(df_clean) < 3:
        return None, None, None
    
    # Create grid with padding
    log_m_min, log_m_max = df_clean['log_moneyness'].quantile([0.05, 0.95])
    tenor_min, tenor_max = df_clean['tenor_days'].quantile([0.05, 0.95])
    
    log_m_range = log_m_max - log_m_min
    tenor_range = tenor_max - tenor_min
    
    log_m_min -= log_m_range * 0.1
    log_m_max += log_m_range * 0.1
    tenor_min = max(1e-6, tenor_min - tenor_range * 0.1)
    tenor_max += tenor_range * 0.1
    
    log_m_grid = np.linspace(log_m_min, log_m_max, grid_resolution)
    tenor_grid = np.linspace(tenor_min, tenor_max, grid_resolution)
    log_m_mesh, tenor_mesh = np.meshgrid(log_m_grid, tenor_grid)
    
    # Interpolate
    points = df_clean[['log_moneyness', 'tenor_days']].values
    values = df_clean['implied_vol'].values
    
    try:
        iv_grid = griddata(
            points, values,
            (log_m_mesh, tenor_mesh),
            method='cubic',
            fill_value=np.nan
        )
    except:
        # Fall back to linear if cubic fails
        iv_grid = griddata(
            points, values,
            (log_m_mesh, tenor_mesh),
            method='linear',
            fill_value=np.nan
        )
    
    return log_m_mesh, tenor_mesh, iv_grid