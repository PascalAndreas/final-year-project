"""
Forward curve recipes for OKX OrderbookStore.

Provides recipes for computing forward curves using:
- PCHIP interpolation with EWMA smoothing
- Kalman-filtered Nelson-Siegel carry model

Usage with functools.partial for configuration:
    from functools import partial
    recipe = partial(build_forwards_pchip, binning='5m', lambda_ewma=0.8)
    lf = store.get_derived(recipe, start, end, cache_name='forwards_pchip_5m')
"""

import polars as pl
import numpy as np
from datetime import datetime, date, timedelta, timezone
from typing import Optional

from forwards.pchip import fit_pchip_curve, ewma_smooth, curves_to_polars, PCHIPCurve
from forwards.kalman_ns import kalman_filter, states_to_polars
from forwards.utils import apply_early_roll_filter, compute_weights


def _finalize_binning(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Finalize binning by promoting time_bin to timeMs and dropping old time columns.
    
    After binning, time_bin contains the aligned timestamps. This function:
    - Drops the original timeMs and exchTimeMs columns
    - Renames time_bin to timeMs
    """
    return lf.drop('timeMs', 'exchTimeMs').rename({'time_bin': 'timeMs'})


def prepare_pillars(
    store,
    inst_family: str,
    dates: list[date],
    binning: Optional[str] = None,
    depth: int = 1,
    min_time_to_expiry_hours: float = 2.0,
    cache_name_suffix: str = "_pillars",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Prepare pillar data from SWAP and FUTURES orderbooks.
    
    Returns (df_swap, df_futures) with early-roll filter applied to futures.
    """
    # Build feature list (ordered for performance: bin early to reduce rows, trim early to reduce columns)
    features = []
    if binning:
        features.append('bin')
    features.extend(['rel_spread', 'trim', 'tenor'])
    if binning:
        features.append(_finalize_binning)
    
    # Shared fetch parameters
    fetch_params = {
        'inst_family': inst_family,
        'dates': dates,
        'depth': depth,
        'binning': binning,
        'features': features,
    }
    
    # Build cache name if binning
    if binning:
        cache_name = f"d{depth}_{binning}{cache_name_suffix}"
    
    # Fetch SWAP (perpetual)
    lf_swap = store.get(
        inst_type='SWAP',
        cache_name=cache_name if binning else None,
        **fetch_params
    )
    
    # Fetch FUTURES
    lf_futures = store.get(
        inst_type='FUTURES',
        cache_name=cache_name if binning else None,
        **fetch_params
    )
    
    # Collect
    df_swap = lf_swap.collect()
    df_futures = lf_futures.collect()
    
    if df_swap.is_empty() or df_futures.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    
    # Apply early-roll filter to futures (drop near-expiry contracts)
    df_futures = apply_early_roll_filter(
        df_futures,
        current_time_col='timeMs',
        expiry_col='expiry',
        min_time_to_expiry_hours=min_time_to_expiry_hours,
    )
    
    return df_swap, df_futures


def build_forwards_pchip(
    store,
    dates: list[date],
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = '5m',
    lambda_ewma: float = 0.8,
    w0_anchor: float = 10.0,
    min_time_to_expiry_hours: float = 2.0,
    max_weight_ratio: float = 100.0,
    unique_times: Optional[list[int]] = None,
) -> pl.LazyFrame:
    """
    Build forward curve using PCHIP interpolation with EWMA smoothing.
    
    Recipe for store.get_derived(). Use functools.partial to configure:
        recipe = partial(build_forwards_pchip, binning='1m', lambda_ewma=0.9)
        lf = store.get_derived(recipe, start, end, cache_name='forwards_pchip_1m')
    
    Args:
        unique_times: Optional list of timeMs values to build curves for.
                     If None and binning is set, uses binned timestamps.
                     If None and no binning, uses union of all timestamps.
    """
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data
    df_swap, df_futures = prepare_pillars(
        store, inst_family, dates, binning, min_time_to_expiry_hours=min_time_to_expiry_hours
    )
    
    if df_swap.is_empty() or df_futures.is_empty():
        return pl.LazyFrame()
    
    # Determine timestamps for curve construction
    if unique_times is not None:
        # Use provided timestamps
        times = sorted(unique_times)
    elif binning:
        # Use binned timestamps (already aligned)
        times = df_swap['timeMs'].unique().sort().to_list()
    else:
        # Union of all timestamps from both dataframes
        times = pl.concat([
            df_swap.select('timeMs'),
            df_futures.select('timeMs')
        ]).unique().sort()['timeMs'].to_list()
    
    # Match timestamps using asof join for snapshot matching
    df_times = pl.DataFrame({'timeMs': times})
    df_swap_matched = df_times.join_asof(df_swap, on='timeMs', strategy='backward')
    df_futures_matched = df_times.join_asof(df_futures, on='timeMs', strategy='backward')
    
    curves = []
    
    for timeMs in times:
        # Get snapshot at this time (single row for swap, multiple rows for futures)
        swap = df_swap_matched.filter(pl.col('timeMs') == timeMs)
        futures = df_futures_matched.filter(pl.col('timeMs') == timeMs)
        
        if swap.is_empty() or futures.is_empty():
            continue
        
        # Check for null values (no match found in asof join)
        if swap['bid_1_px'][0] is None or futures['bid_1_px'][0] is None:
            continue
        
        # Extract swap anchor
        F_bid_swap = swap['bid_1_px'][0]
        F_ask_swap = swap['ask_1_px'][0]
        
        # Extract futures pillars
        T_pillars = futures['T'].to_numpy()
        F_bid_pillars = futures['bid_1_px'].to_numpy()
        F_ask_pillars = futures['ask_1_px'].to_numpy()
        symbols = futures['symbol'].to_list()
        rel_spreads = futures['rel_spread'].to_numpy()
        
        # Compute weights
        weights = compute_weights(rel_spreads, max_weight_ratio=max_weight_ratio)
        
        # Fit PCHIP curve
        curve = fit_pchip_curve(
            T_pillars=T_pillars,
            F_bid_pillars=F_bid_pillars,
            F_ask_pillars=F_ask_pillars,
            symbols=symbols,
            timeMs=int(timeMs),
            T_swap=0.0,
            F_bid_swap=F_bid_swap,
            F_ask_swap=F_ask_swap,
            w0_anchor=w0_anchor,
            weights=weights,
        )
        curves.append(curve)
    
    if not curves:
        return pl.LazyFrame()
    
    # Apply EWMA smoothing
    smoothed_curves = ewma_smooth(curves, lambda_ewma=lambda_ewma)
    
    # Convert to Polars
    df_result = curves_to_polars(smoothed_curves)
    
    return df_result.lazy()


def build_forwards_kalman(
    store,
    dates: list[date],
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = '1m',
    lambda_ns: float = 0.1,
    process_noise_scale: float = 1e-4,
    ar1_coef: float = 0.99,
    min_time_to_expiry_hours: float = 2.0,
    spread_to_variance_scale: float = 1.0,
    unique_times: Optional[list[int]] = None,
) -> pl.LazyFrame:
    """
    Build forward curve using Kalman-filtered Nelson-Siegel carry model.
    
    Recipe for store.get_derived(). Use functools.partial to configure:
        recipe = partial(build_forwards_kalman, binning='5m', lambda_ns=0.05)
        lf = store.get_derived(recipe, start, end, cache_name='forwards_kalman_5m')
    
    Args:
        unique_times: Optional list of timeMs values to build curves for.
                     If None and binning is set, uses binned timestamps.
                     If None and no binning, uses union of all timestamps.
    """
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data
    df_swap, df_futures = prepare_pillars(
        store, inst_family, dates, binning, min_time_to_expiry_hours=min_time_to_expiry_hours
    )
    
    if df_swap.is_empty() or df_futures.is_empty():
        return pl.LazyFrame()
    
    # Determine timestamps for curve construction
    if unique_times is not None:
        # Use provided timestamps
        times = sorted(unique_times)
    elif binning:
        # Use binned timestamps (already aligned)
        times = df_swap['timeMs'].unique().sort().to_list()
    else:
        # Union of all timestamps from both dataframes
        times = pl.concat([
            df_swap.select('timeMs'),
            df_futures.select('timeMs')
        ]).unique().sort()['timeMs'].to_list()
    
    # Match timestamps using asof join for snapshot matching
    df_times = pl.DataFrame({'timeMs': times})
    df_swap_matched = df_times.join_asof(df_swap, on='timeMs', strategy='backward')
    df_futures_matched = df_times.join_asof(df_futures, on='timeMs', strategy='backward')
    
    snapshots = []
    
    for timeMs in times:
        # Get snapshot at this time
        swap = df_swap_matched.filter(pl.col('timeMs') == timeMs)
        futures = df_futures_matched.filter(pl.col('timeMs') == timeMs)
        
        if swap.is_empty() or futures.is_empty():
            continue
        
        # Check for null values (no match found in asof join)
        if swap['bid_1_px'][0] is None or futures['bid_1_px'][0] is None:
            continue
        
        # Extract swap reference
        F_ref_bid = swap['bid_1_px'][0]
        F_ref_ask = swap['ask_1_px'][0]
        
        # Extract futures pillars
        T_pillars = futures['T'].to_numpy()
        F_bid_pillars = futures['bid_1_px'].to_numpy()
        F_ask_pillars = futures['ask_1_px'].to_numpy()
        spreads = futures['spread'].to_numpy()
        
        snapshots.append({
            'timeMs': int(timeMs),
            'T_pillars': T_pillars,
            'F_bid_pillars': F_bid_pillars,
            'F_ask_pillars': F_ask_pillars,
            'spreads': spreads,
            'F_ref_bid': F_ref_bid,
            'F_ref_ask': F_ref_ask,
        })
    
    if not snapshots:
        return pl.LazyFrame()
    
    # Apply Kalman filter
    states = kalman_filter(
        snapshots=snapshots,
        lambda_ns=lambda_ns,
        process_noise_scale=process_noise_scale,
        ar1_coef=ar1_coef,
        spread_to_variance_scale=spread_to_variance_scale,
    )
    
    # Convert to Polars
    df_result = states_to_polars(states)
    
    return df_result.lazy()

