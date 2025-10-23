"""
Implied Volatility Surface Visualization Module

Concise implementation for preparing orderbook data and creating 
interactive IV surface plots with time slider.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from tqdm.auto import tqdm
import plotly.graph_objects as go

from .black import black76_implied_volatility
from .helpers import parse_option_instrument


def prepare_orderbook_features(orderbook_df: pd.DataFrame, future_price: float) -> pd.DataFrame:
    """
    Prepare orderbook data with features needed for IV calculation.
    
    Parameters:
    -----------
    orderbook_df : pd.DataFrame
        Orderbook data with bid_1_px, ask_1_px, etc.
    future_price : float
        Forward/future price for the underlying
    
    Returns:
    --------
    pd.DataFrame : DataFrame with columns:
        - timestamp: timeMs from orderbook
        - instrument: symbol/instrument name
        - bid: best bid price
        - ask: best ask price
        - mid: mid price (bid + ask) / 2
        - spread: ask - bid
        - spread_pct: spread / mid * 100
        - forward_price: future price
    """
    df = orderbook_df.copy()
    
    # Extract key features
    features = pd.DataFrame({
        'timestamp': df['timeMs'],
        'instrument': df['symbol'],
        'bid': df['bid_1_px'],
        'ask': df['ask_1_px'],
        'forward_price': future_price
    })
    
    # Calculate derived features
    features['mid'] = (features['bid'] + features['ask']) / 2
    features['spread'] = features['ask'] - features['bid']
    features['spread_pct'] = (features['spread'] / features['mid'] * 100)
    
    return features


def compute_iv_surface(features_df: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Compute implied volatility surface from orderbook features.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Output from prepare_orderbook_features with mid prices
    risk_free_rate : float
        Risk-free rate (default 0 for crypto)
    
    Returns:
    --------
    pd.DataFrame : IV surface data with columns:
        - timestamp
        - instrument
        - expiry_date
        - strike
        - option_type (C/P)
        - mid_price
        - forward_price
        - tenor_years
        - log_moneyness: ln(K/F)
        - implied_vol
    """
    results = []
    
    for _, row in features_df.iterrows():
        try:
            # Parse instrument
            inst_family, expiry_date, strike, option_type = parse_option_instrument(row['instrument'])
            
            # Calculate time to expiry
            if isinstance(row['timestamp'], (int, float)):
                current_time = pd.Timestamp(row['timestamp'], unit='ms')
            else:
                current_time = pd.Timestamp(row['timestamp'])
            
            tenor_years = (expiry_date - current_time).total_seconds() / (365.25 * 24 * 3600)
            
            if tenor_years <= 0:
                continue
            
            # Calculate log moneyness
            log_moneyness = np.log(strike / row['forward_price'])
            
            # Compute implied volatility
            iv = black76_implied_volatility(
                market_price=row['mid'],
                F=row['forward_price'],
                K=strike,
                T=tenor_years,
                r=risk_free_rate,
                option_type=option_type
            )
            
            results.append({
                'timestamp': row['timestamp'],
                'instrument': row['instrument'],
                'expiry_date': expiry_date,
                'strike': strike,
                'option_type': option_type,
                'mid_price': row['mid'],
                'forward_price': row['forward_price'],
                'tenor_years': tenor_years,
                'log_moneyness': log_moneyness,
                'implied_vol': iv
            })
            
        except (ValueError, IndexError, ZeroDivisionError):
            continue
    
    return pd.DataFrame(results)


def create_iv_surface_plot(
    iv_df: pd.DataFrame,
    time_col: str = 'timestamp',
    time_step_minutes: int = 5,
    grid_resolution: int = 40,
    title: str = "Implied Volatility Surface"
) -> go.Figure:
    """
    Create interactive IV surface plot with time slider.
    
    Parameters:
    -----------
    iv_df : pd.DataFrame
        IV surface data from compute_iv_surface
    time_col : str
        Column name for timestamp
    time_step_minutes : int
        Time step for slider frames (minutes)
    grid_resolution : int
        Grid resolution for surface interpolation
    title : str
        Plot title
    
    Returns:
    --------
    go.Figure : Plotly figure with time slider
    """
    # Convert timestamps to datetime
    if iv_df[time_col].dtype in ['int64', 'float64']:
        iv_df['datetime'] = pd.to_datetime(iv_df[time_col], unit='ms')
    else:
        iv_df['datetime'] = pd.to_datetime(iv_df[time_col])
    
    # Filter valid IVs
    iv_df = iv_df[np.isfinite(iv_df['implied_vol']) & (iv_df['implied_vol'] > 0)]
    
    if len(iv_df) == 0:
        print("No valid IV data to plot")
        return go.Figure()
    
    # Create time bins
    min_time = iv_df['datetime'].min()
    max_time = iv_df['datetime'].max()
    time_bins = pd.date_range(
        start=min_time.floor(f'{time_step_minutes}min'),
        end=max_time.ceil(f'{time_step_minutes}min'),
        freq=f'{time_step_minutes}min'
    )
    
    # Assign each row to a time bin
    iv_df['time_bin'] = pd.cut(iv_df['datetime'], bins=time_bins, labels=time_bins[:-1])
    
    # Create frames for each time bin
    frames = []
    z_min, z_max = iv_df['implied_vol'].quantile([0.01, 0.99])
    
    for time_bin in tqdm(time_bins[:-1], desc="Creating frames"):
        bin_data = iv_df[iv_df['time_bin'] == time_bin]
        
        if len(bin_data) < 3:
            # Not enough data for this frame
            frames.append(go.Frame(
                data=[go.Surface(x=[], y=[], z=[], showscale=False)],
                name=str(time_bin)
            ))
            continue
        
        # Interpolate onto grid
        log_m_grid, tenor_grid, iv_grid = _interpolate_surface(
            bin_data, grid_resolution
        )
        
        if log_m_grid is None:
            frames.append(go.Frame(
                data=[go.Surface(x=[], y=[], z=[], showscale=False)],
                name=str(time_bin)
            ))
            continue
        
        # Create surface trace
        surface = go.Surface(
            x=log_m_grid,
            y=tenor_grid * 365,  # Convert to days
            z=iv_grid,
            colorscale='Viridis',
            cmin=z_min,
            cmax=z_max,
            colorbar=dict(title="IV", x=1.02),
            showscale=True
        )
        
        frames.append(go.Frame(data=[surface], name=str(time_bin)))
    
    # Create initial figure with first frame
    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Log Moneyness ln(K/F)',
            yaxis_title='Tenor (days)',
            zaxis_title='Implied Volatility',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 0
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'y': 0,
            'xanchor': 'left',
            'x': 0.3,
            'currentvalue': {
                'prefix': 'Time: ',
                'visible': True,
                'xanchor': 'right'
            },
            'steps': [
                {
                    'args': [[frame.name], {
                        'frame': {'duration': 300, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 300}
                    }],
                    'method': 'animate',
                    'label': str(frame.name)[:16]  # Truncate for display
                }
                for frame in frames
            ]
        }],
        height=700,
        width=1000
    )
    
    return fig


def _interpolate_surface(df: pd.DataFrame, grid_resolution: int = 40):
    """
    Interpolate sparse IV data onto a regular grid.
    
    Returns: log_moneyness_grid, tenor_grid, iv_grid (2D arrays)
    """
    if len(df) < 3:
        return None, None, None
    
    # Remove NaN/inf values
    df_clean = df[
        np.isfinite(df['log_moneyness']) & 
        np.isfinite(df['tenor_years']) & 
        np.isfinite(df['implied_vol']) &
        (df['tenor_years'] > 0)
    ].copy()
    
    if len(df_clean) < 3:
        return None, None, None
    
    # Create grid with padding
    log_m_min, log_m_max = df_clean['log_moneyness'].quantile([0.05, 0.95])
    tenor_min, tenor_max = df_clean['tenor_years'].quantile([0.05, 0.95])
    
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
    points = df_clean[['log_moneyness', 'tenor_years']].values
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

