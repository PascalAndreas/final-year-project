"""
IV Surface Visualization Module

Provides interactive 3D volatility surface plotting with time slider,
using Black-76 model for implied volatility calculation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from tqdm.auto import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .black import black76_implied_volatility
from .helpers import moneyness as calc_moneyness, parse_option_instrument


def log_moneyness(K, F):
    """Calculate log-moneyness: ln(K/F)"""
    if F <= 0:
        return np.nan
    return np.log(K / F)


def get_futures_prices_at_time(futures_df, timestamp, lookback_seconds=21600):
    """
    Get the most recent futures prices for each instrument at a given timestamp.
    
    Parameters:
    - futures_df: DataFrame with futures trades
    - timestamp: Target timestamp in milliseconds
    - lookback_seconds: Maximum age of futures price (default 6 hours)
    
    Returns:
    - dict: {instrument_name: (mark_price, age_seconds)}
    """
    lookback_ms = lookback_seconds * 1000
    min_timestamp = timestamp - lookback_ms
    
    # Get all futures trades within lookback window
    recent_trades = futures_df[
        (futures_df['timestamp'] <= timestamp) & 
        (futures_df['timestamp'] >= min_timestamp)
    ]
    
    if len(recent_trades) == 0:
        return {}
    
    # Get the most recent trade for each instrument
    latest_trades = recent_trades.sort_values('timestamp').groupby('instrument_name').last()
    
    # Calculate age in seconds
    result = {}
    for instrument, row in latest_trades.iterrows():
        age_seconds = (timestamp - row['timestamp']) / 1000
        result[instrument] = (row['mark_price'], age_seconds)
    
    return result


def compute_iv_at_timestamp(options_df, futures_df, timestamp, 
                            lookback_seconds=21600, risk_free_rate=0.0):
    """
    Compute implied volatilities for all options at a specific timestamp.
    
    Parameters:
    - options_df: DataFrame with option trades
    - futures_df: DataFrame with futures trades
    - timestamp: Target timestamp in milliseconds
    - lookback_seconds: Maximum age for marks (default 6 hours)
    - risk_free_rate: Risk-free rate (default 0 for crypto)
    
    Returns:
    - DataFrame with columns: expiry, strike, option_type, mark_price, forward_price,
                              log_moneyness, tenor_years, implied_vol, mark_age_seconds
    """
    lookback_ms = lookback_seconds * 1000
    min_timestamp = timestamp - lookback_ms
    
    # Get futures prices
    futures_prices = get_futures_prices_at_time(futures_df, timestamp, lookback_seconds)
    
    if len(futures_prices) == 0:
        return pd.DataFrame()
    
    # Get recent option trades within lookback window
    recent_options = options_df[
        (options_df['timestamp'] <= timestamp) & 
        (options_df['timestamp'] >= min_timestamp)
    ]
    
    if len(recent_options) == 0:
        return pd.DataFrame()
    
    # Get most recent trade for each unique option
    latest_options = recent_options.sort_values('timestamp').groupby('instrument_name').last()
    
    results = []
    current_date = datetime.fromtimestamp(timestamp / 1000)
    
    for instrument, row in latest_options.iterrows():
        expiry, strike, opt_type = parse_option_instrument(instrument)
        if expiry is None or strike is None:
            continue
        
        # Get corresponding future
        future_instrument = f"BTC-{expiry}"
        if future_instrument not in futures_prices:
            continue
        
        forward_price, future_age = futures_prices[future_instrument]
        
        # Calculate time to expiry (in years)
        try:
            expiry_date = datetime.strptime(expiry, '%d%b%y')
        except ValueError:
            continue
            
        time_to_expiry = (expiry_date - current_date).total_seconds() / (365.25 * 24 * 3600)
        
        if time_to_expiry <= 0:
            continue
        
        # Calculate mark age
        mark_age_seconds = (timestamp - row['timestamp']) / 1000
        
        # Calculate implied volatility
        iv = black76_implied_volatility(
            row['mark_price'], 
            forward_price, 
            strike, 
            time_to_expiry, 
            risk_free_rate, 
            opt_type
        )
        
        if not np.isnan(iv) and iv > 0:
            results.append({
                'expiry': expiry,
                'strike': strike,
                'option_type': opt_type,
                'mark_price': row['mark_price'],
                'forward_price': forward_price,
                'log_moneyness': log_moneyness(strike, forward_price),
                'tenor_years': time_to_expiry,
                'implied_vol': iv,
                'mark_age_seconds': mark_age_seconds,
                'future_age_seconds': future_age
            })
    
    return pd.DataFrame(results)


def prepare_iv_timeseries(options_df, futures_df, time_step_minutes=5, 
                          lookback_seconds=21600, risk_free_rate=0.0):
    """
    Pre-compute IV surfaces for a series of timestamps.
    
    Parameters:
    - options_df: DataFrame with option trades
    - futures_df: DataFrame with futures trades  
    - time_step_minutes: Minutes between each surface snapshot
    - lookback_seconds: Maximum age for marks (default 6 hours)
    - risk_free_rate: Risk-free rate (default 0 for crypto)
    
    Returns:
    - dict: {timestamp: DataFrame of IVs at that timestamp}
    """
    # Get time range
    start_time = options_df['timestamp'].min()
    end_time = options_df['timestamp'].max()
    
    # Create timestamps at regular intervals
    time_step_ms = time_step_minutes * 60 * 1000
    timestamps = np.arange(start_time, end_time + time_step_ms, time_step_ms)
    
    print(f"Computing IV surfaces for {len(timestamps)} timestamps...")
    
    iv_timeseries = {}
    for ts in tqdm(timestamps, desc="Building IV timeseries"):
        iv_df = compute_iv_at_timestamp(
            options_df, futures_df, ts, 
            lookback_seconds, risk_free_rate
        )
        if len(iv_df) > 0:
            iv_timeseries[ts] = iv_df
    
    print(f"✓ Computed {len(iv_timeseries)} valid IV surfaces")
    return iv_timeseries


def get_trades_window(options_df, timestamp, window_minutes=10):
    """
    Get all trades within a time window around a timestamp.
    
    Parameters:
    - options_df: DataFrame with option trades
    - timestamp: Center timestamp in milliseconds
    - window_minutes: Window size (±minutes)
    
    Returns:
    - DataFrame of trades with computed fields
    """
    window_ms = window_minutes * 60 * 1000
    min_ts = timestamp - window_ms
    max_ts = timestamp + window_ms
    
    trades = options_df[
        (options_df['timestamp'] >= min_ts) & 
        (options_df['timestamp'] <= max_ts)
    ].copy()
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    # Parse instrument info
    instrument_info = trades['instrument_name'].apply(parse_option_instrument)
    trades['expiry'] = instrument_info.apply(lambda x: x[0])
    trades['strike'] = instrument_info.apply(lambda x: x[1])
    trades['option_type'] = instrument_info.apply(lambda x: x[2])
    
    # Calculate time offset from center
    trades['time_offset_seconds'] = (trades['timestamp'] - timestamp) / 1000
    
    return trades


def interpolate_surface(df, grid_resolution=50):
    """
    Interpolate sparse IV data onto a regular grid.
    
    Parameters:
    - df: DataFrame with log_moneyness, tenor_years, implied_vol
    - grid_resolution: Number of grid points in each dimension
    
    Returns:
    - log_moneyness_grid, tenor_grid, iv_grid (2D arrays)
    """
    if len(df) < 3:  # Need at least 3 points to interpolate
        return None, None, None
    
    # Remove any NaN or infinite values
    df_clean = df[
        np.isfinite(df['log_moneyness']) & 
        np.isfinite(df['tenor_years']) & 
        np.isfinite(df['implied_vol'])
    ]
    
    if len(df_clean) < 3:
        return None, None, None
    
    # Create grid
    log_m_min, log_m_max = df_clean['log_moneyness'].min(), df_clean['log_moneyness'].max()
    tenor_min, tenor_max = df_clean['tenor_years'].min(), df_clean['tenor_years'].max()
    
    # Add some padding
    log_m_range = log_m_max - log_m_min
    tenor_range = tenor_max - tenor_min
    
    log_m_min -= log_m_range * 0.1
    log_m_max += log_m_range * 0.1
    tenor_min = max(0, tenor_min - tenor_range * 0.1)
    tenor_max += tenor_range * 0.1
    
    log_moneyness_grid = np.linspace(log_m_min, log_m_max, grid_resolution)
    tenor_grid = np.linspace(tenor_min, tenor_max, grid_resolution)
    log_moneyness_mesh, tenor_mesh = np.meshgrid(log_moneyness_grid, tenor_grid)
    
    # Interpolate
    points = df_clean[['log_moneyness', 'tenor_years']].values
    values = df_clean['implied_vol'].values
    
    iv_grid = griddata(
        points, values, 
        (log_moneyness_mesh, tenor_mesh), 
        method='cubic',
        fill_value=np.nan
    )
    
    return log_moneyness_mesh, tenor_mesh, iv_grid


def create_surface_trace(df, name, colorscale='Viridis', showscale=True, opacity=0.8, scene='scene'):
    """Create a Plotly surface trace from IV data."""
    log_m_grid, tenor_grid, iv_grid = interpolate_surface(df)
    
    if log_m_grid is None:
        # Not enough data, return scatter instead
        return go.Scatter3d(
            x=df['log_moneyness'],
            y=np.log10(df['tenor_years'] * 365),  # Log scale tenor in days
            z=df['implied_vol'],
            mode='markers',
            marker=dict(size=3, color=df['implied_vol'], colorscale=colorscale),
            name=name,
            showlegend=True,
            scene=scene
        )
    
    return go.Surface(
        x=log_m_grid,
        y=np.log10(tenor_grid * 365),  # Log scale tenor in days
        z=iv_grid,
        colorscale=colorscale,
        name=name,
        showscale=showscale,
        opacity=opacity,
        colorbar=dict(title="IV", x=1.1) if showscale else None,
        scene=scene
    )


def create_scatter_trace(df, name, color_col='implied_vol', size=3, colorscale='Viridis', scene='scene'):
    """Create a Plotly scatter trace for actual data points."""
    # Color by staleness (age of mark)
    if 'mark_age_seconds' in df.columns:
        # Normalize age to 0-1 (0=fresh, 1=6hr old)
        max_age = 6 * 3600
        color_values = np.clip(df['mark_age_seconds'] / max_age, 0, 1)
        text = [f"IV: {iv:.2%}<br>Age: {age/60:.1f}min<br>Strike: {strike}<br>Expiry: {exp}<br>Tenor: {tenor*365:.1f} days" 
                for iv, age, strike, exp, tenor in zip(
                    df['implied_vol'], df['mark_age_seconds'], 
                    df['strike'], df['expiry'], df['tenor_years']
                )]
    else:
        color_values = df[color_col]
        text = [f"IV: {iv:.2%}<br>Strike: {strike}<br>Expiry: {exp}<br>Tenor: {tenor*365:.1f} days" 
                for iv, strike, exp, tenor in zip(df['implied_vol'], df['strike'], df['expiry'], df['tenor_years'])]
    
    return go.Scatter3d(
        x=df['log_moneyness'],
        y=np.log10(df['tenor_years'] * 365),  # Log scale tenor
        z=df['implied_vol'],
        mode='markers',
        marker=dict(
            size=size,
            color=color_values,
            colorscale=colorscale,
            showscale=False,
            line=dict(width=0.5, color='white')
        ),
        name=name,
        text=text,
        hovertemplate='%{text}<br>Log-moneyness: %{x:.3f}<extra></extra>',
        showlegend=True,
        scene=scene
    )


def create_trade_scatter_trace(trades_df, futures_df, timestamp, name, 
                               color_map={'buy': 'green', 'sell': 'red'}, scene='scene'):
    """Create scatter trace for recent trades."""
    if len(trades_df) == 0:
        return None
    
    # Get futures prices at this timestamp
    futures_prices = get_futures_prices_at_time(futures_df, timestamp, lookback_seconds=21600)
    
    # Calculate IVs and positions for trades
    results = []
    current_date = datetime.fromtimestamp(timestamp / 1000)
    
    for _, row in trades_df.iterrows():
        expiry = row['expiry']
        if expiry is None:
            continue
            
        future_instrument = f"BTC-{expiry}"
        if future_instrument not in futures_prices:
            continue
        
        forward_price, _ = futures_prices[future_instrument]
        
        try:
            expiry_date = datetime.strptime(expiry, '%d%b%y')
        except ValueError:
            continue
            
        tenor_years = (expiry_date - current_date).total_seconds() / (365.25 * 24 * 3600)
        if tenor_years <= 0:
            continue
        
        # Calculate IV for this trade
        iv = black76_implied_volatility(
            row['price'],  # Use trade price, not mark
            forward_price,
            row['strike'],
            tenor_years,
            0.0,
            row['option_type']
        )
        
        if not np.isnan(iv) and iv > 0:
            results.append({
                'log_moneyness': log_moneyness(row['strike'], forward_price),
                'tenor_years': tenor_years,
                'implied_vol': iv,
                'direction': row['direction'],
                'amount': row['amount'],
                'price': row['price'],
                'strike': row['strike'],
                'expiry': expiry,
                'time_offset': row['time_offset_seconds']
            })
    
    if len(results) == 0:
        return None
    
    trade_df = pd.DataFrame(results)
    
    # Create color array based on direction
    colors = [color_map.get(d, 'gray') for d in trade_df['direction']]
    
    # Size by volume
    sizes = np.clip(trade_df['amount'] / 10, 3, 15)
    
    text = [f"Trade: {dir}<br>IV: {iv:.2%}<br>Price: {price:.4f}<br>Amount: {amt}<br>"
            f"Strike: {strike}<br>Tenor: {tenor*365:.1f} days<br>Time: {offset:+.0f}s" 
            for dir, iv, price, amt, strike, tenor, offset in zip(
                trade_df['direction'], trade_df['implied_vol'], trade_df['price'],
                trade_df['amount'], trade_df['strike'], trade_df['tenor_years'], trade_df['time_offset']
            )]
    
    return go.Scatter3d(
        x=trade_df['log_moneyness'],
        y=np.log10(trade_df['tenor_years'] * 365),  # Log scale tenor
        z=trade_df['implied_vol'],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        name=name,
        text=text,
        hovertemplate='%{text}<extra></extra>',
        showlegend=True,
        scene=scene
    )


def create_iv_surface_viewer(iv_timeseries, options_df, futures_df, 
                             trade_window_minutes=10, show_trades=True):
    """
    Create interactive IV surface viewer with time slider.
    
    Parameters:
    - iv_timeseries: dict from prepare_iv_timeseries
    - options_df: DataFrame with option trades
    - futures_df: DataFrame with futures trades
    - trade_window_minutes: Window for showing recent trades
    - show_trades: Whether to show trade scatter
    
    Returns:
    - plotly Figure object
    """
    if len(iv_timeseries) == 0:
        print("No IV data to plot")
        return None
    
    timestamps = sorted(iv_timeseries.keys())
    
    # Calculate global bounds across all timestamps to prevent jumpy axes
    print("Computing global axis bounds...")
    all_log_moneyness = []
    all_tenor_years = []
    all_iv = []
    
    for ts in timestamps:
        iv_df = iv_timeseries[ts]
        all_log_moneyness.extend(iv_df['log_moneyness'].dropna().tolist())
        all_tenor_years.extend(iv_df['tenor_years'].dropna().tolist())
        all_iv.extend(iv_df['implied_vol'].dropna().tolist())
    
    # Set fixed ranges with some padding
    log_m_min, log_m_max = np.percentile(all_log_moneyness, [1, 99])
    log_m_range = log_m_max - log_m_min
    log_m_min -= log_m_range * 0.1
    log_m_max += log_m_range * 0.1
    
    tenor_min, tenor_max = np.percentile(all_tenor_years, [1, 99])
    # Use log scale for tenor (standard practice for IV surfaces)
    log_tenor_min = np.log10(max(tenor_min, 1/365))  # Minimum 1 day
    log_tenor_max = np.log10(tenor_max)
    
    iv_min, iv_max = np.percentile(all_iv, [1, 99])
    iv_range = iv_max - iv_min
    iv_min = max(0, iv_min - iv_range * 0.1)
    iv_max += iv_range * 0.1
    
    print(f"  Log-moneyness range: [{log_m_min:.3f}, {log_m_max:.3f}]")
    print(f"  Tenor range: [{tenor_min*365:.1f}, {tenor_max*365:.1f}] days")
    print(f"  IV range: [{iv_min:.2%}, {iv_max:.2%}]")
    
    # Create subplots for calls and puts
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Call Options', 'Put Options'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.05
    )
    
    # Create frames for animation
    frames = []
    
    print(f"Creating visualization frames for {len(timestamps)} timestamps...")
    
    for ts in tqdm(timestamps, desc="Building frames"):
        iv_df = iv_timeseries[ts]
        
        # Split by option type
        calls_df = iv_df[iv_df['option_type'] == 'C']
        puts_df = iv_df[iv_df['option_type'] == 'P']
        
        frame_data = []
        
        # Call surface (scene)
        if len(calls_df) > 0:
            call_surface = create_surface_trace(
                calls_df, 
                'Call Surface',
                colorscale='Blues',
                showscale=False,
                opacity=0.7,
                scene='scene'
            )
            frame_data.append(call_surface)
            
            # Call data points
            call_scatter = create_scatter_trace(
                calls_df,
                'Call Marks',
                size=4,
                colorscale='RdYlGn_r',  # Red=stale, Green=fresh
                scene='scene'
            )
            frame_data.append(call_scatter)
        
        # Put surface (scene2)
        if len(puts_df) > 0:
            put_surface = create_surface_trace(
                puts_df,
                'Put Surface', 
                colorscale='Reds',
                showscale=False,
                opacity=0.7,
                scene='scene2'
            )
            frame_data.append(put_surface)
            
            # Put data points
            put_scatter = create_scatter_trace(
                puts_df,
                'Put Marks',
                size=4,
                colorscale='RdYlGn_r',
                scene='scene2'
            )
            frame_data.append(put_scatter)
        
        # Add recent trades - show on both subplots
        if show_trades:
            trades = get_trades_window(options_df, ts, trade_window_minutes)
            if len(trades) > 0:
                # Trades on call subplot
                trade_trace_calls = create_trade_scatter_trace(
                    trades, futures_df, ts, 'Recent Trades (C)', scene='scene'
                )
                if trade_trace_calls is not None:
                    frame_data.append(trade_trace_calls)
                
                # Trades on put subplot
                trade_trace_puts = create_trade_scatter_trace(
                    trades, futures_df, ts, 'Recent Trades (P)', scene='scene2'
                )
                if trade_trace_puts is not None:
                    frame_data.append(trade_trace_puts)
        
        dt = datetime.fromtimestamp(ts / 1000)
        frames.append(go.Frame(
            data=frame_data,
            name=str(ts),
            layout=dict(title=f'IV Surface - {dt.strftime("%Y-%m-%d %H:%M:%S")}')
        ))
    
    # Set initial frame
    initial_ts = timestamps[len(timestamps) // 2]  # Start in the middle
    initial_iv = iv_timeseries[initial_ts]
    initial_calls = initial_iv[initial_iv['option_type'] == 'C']
    initial_puts = initial_iv[initial_iv['option_type'] == 'P']
    
    # Add initial traces with proper scene assignment
    # Note: Don't use row/col with 3D scenes - the scene parameter handles subplot assignment
    if len(initial_calls) > 0:
        fig.add_trace(create_surface_trace(initial_calls, 'Call Surface', 'Blues', False, 0.7, scene='scene'))
        fig.add_trace(create_scatter_trace(initial_calls, 'Call Marks', size=4, colorscale='RdYlGn_r', scene='scene'))
    
    if len(initial_puts) > 0:
        fig.add_trace(create_surface_trace(initial_puts, 'Put Surface', 'Reds', False, 0.7, scene='scene2'))
        fig.add_trace(create_scatter_trace(initial_puts, 'Put Marks', size=4, colorscale='RdYlGn_r', scene='scene2'))
    
    if show_trades:
        initial_trades = get_trades_window(options_df, initial_ts, trade_window_minutes)
        if len(initial_trades) > 0:
            trade_trace_calls = create_trade_scatter_trace(initial_trades, futures_df, initial_ts, 'Recent Trades (C)', scene='scene')
            if trade_trace_calls is not None:
                fig.add_trace(trade_trace_calls)
            # Duplicate for puts subplot
            trade_trace_puts = create_trade_scatter_trace(initial_trades, futures_df, initial_ts, 'Recent Trades (P)', scene='scene2')
            if trade_trace_puts is not None:
                fig.add_trace(trade_trace_puts)
    
    # Add frames
    fig.frames = frames
    
    # Create slider
    sliders = [dict(
        active=len(timestamps) // 2,
        yanchor="top",
        y=0,
        xanchor="left",
        x=0.05,
        currentvalue=dict(
            prefix="Time: ",
            visible=True,
            xanchor="left"
        ),
        pad=dict(b=10, t=50),
        len=0.9,
        steps=[dict(
            args=[[str(ts)], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
                transition=dict(duration=0)
            )],
            method="animate",
            label=datetime.fromtimestamp(ts / 1000).strftime("%m-%d %H:%M")
        ) for ts in timestamps]
    )]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Bitcoin Options Implied Volatility Surface<br><sub>Mark age colored (green=fresh, red=stale). Hard cutoff: 6 hours</sub>',
            x=0.5,
            xanchor='center'
        ),
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.05,
            y=1.15,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=200, redraw=True),
                                     fromcurrent=True,
                                     mode='immediate',
                                     transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate",
                                       transition=dict(duration=0))])
            ]
        )],
        height=700,
        showlegend=True,
        legend=dict(x=0.5, y=1.0, xanchor='center', yanchor='bottom', orientation='h')
    )
    
    # Update 3D scene settings for both subplots with fixed ranges
    # Create custom tickvals and ticktext for log tenor axis
    tenor_tick_days = [1, 2, 5, 10, 20, 50, 100, 200, 365]
    tenor_tickvals = [np.log10(d) for d in tenor_tick_days if log_tenor_min <= np.log10(d) <= log_tenor_max]
    tenor_ticktext = [f"{d}d" if d < 365 else f"{d/365:.1f}y" for d in tenor_tick_days if log_tenor_min <= np.log10(d) <= log_tenor_max]
    
    scene_config = dict(
        xaxis=dict(
            title="Log-Moneyness ln(K/F)",
            range=[log_m_min, log_m_max]
        ),
        yaxis=dict(
            title="Tenor (log scale)",
            range=[log_tenor_min, log_tenor_max],
            tickvals=tenor_tickvals,
            ticktext=tenor_ticktext
        ),
        zaxis=dict(
            title="Implied Volatility",
            range=[iv_min, iv_max]
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        )
    )
    
    fig.update_scenes(scene_config)
    
    print("✓ IV surface viewer created")
    return fig

