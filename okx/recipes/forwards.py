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
from forwards.utils import add_expiry_and_ttm, apply_early_roll_filter, compute_weights


def prepare_pillars(
    store,
    inst_family: str,
    dates: list[date],
    binning: Optional[str] = None,
    depth: int = 1,
    min_time_to_expiry_hours: float = 2.0,
    cache_name_suffix: str = "_pillars",
) -> pl.DataFrame:
    """
    Prepare pillar data from SWAP and FUTURES orderbooks.
    
    Fetches SWAP and FUTURES data, applies early-roll filter, and combines.
    """
    # Shared fetch parameters
    fetch_params = {
        'inst_family': inst_family,
        'dates': dates,
        'depth': depth,
        'binning': binning,
        'features': ['tenor', 'inst_type', 'rel_spread'],
    }
    
    # Build cache name if binning
    if binning:
        cache_name = f"d{depth}_{binning}{cache_name_suffix}"
    else:
        cache_name = f"d{depth}{cache_name_suffix}"
    
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
        return pl.DataFrame()
    
    # Apply early-roll filter to futures (drop near-expiry contracts)
    df_futures = apply_early_roll_filter(
        df_futures,
        current_time_col='timeMs',
        expiry_col='expiry',
        min_time_to_expiry_hours=min_time_to_expiry_hours,
    )
    
    if df_futures.is_empty():
        return pl.DataFrame()  # All futures filtered out
    
    # Align timeMs when binning: use max timeMs per time_bin
    if binning:
        max_time_per_bin = pl.concat([df_swap, df_futures]).group_by('time_bin').agg(
            pl.col('timeMs').max().alias('timeMs_aligned')
        )
        df_swap = df_swap.join(max_time_per_bin, on='time_bin').with_columns(
            pl.col('timeMs_aligned').alias('timeMs')
        ).drop('timeMs_aligned')
        df_futures = df_futures.join(max_time_per_bin, on='time_bin').with_columns(
            pl.col('timeMs_aligned').alias('timeMs')
        ).drop('timeMs_aligned')
    
    # Combine (separate rows, distinguished by inst_type)
    return pl.concat([df_swap, df_futures], how="diagonal").sort('timeMs')


def build_forwards_pchip(
    store,
    dates: list[date],
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = '5m',
    lambda_ewma: float = 0.8,
    w0_anchor: float = 10.0,
    min_time_to_expiry_hours: float = 2.0,
    max_weight_ratio: float = 100.0,
) -> pl.LazyFrame:
    """
    Build forward curve using PCHIP interpolation with EWMA smoothing.
    
    Recipe for store.get_derived(). Use functools.partial to configure:
        recipe = partial(build_forwards_pchip, binning='1m', lambda_ewma=0.9)
        lf = store.get_derived(recipe, start, end, cache_name='forwards_pchip_1m')
    """
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data
    df_pillars = prepare_pillars(
        store, inst_family, dates, binning, min_time_to_expiry_hours=min_time_to_expiry_hours
    )
    
    if df_pillars.is_empty():
        return pl.LazyFrame()
    
    # Group by timeMs and fit curves
    unique_times = df_pillars['timeMs'].unique().sort()
    curves = []
    
    for timeMs in unique_times:
        snapshot = df_pillars.filter(pl.col('timeMs') == timeMs)
        
        # Separate swap and futures
        swap = snapshot.filter(pl.col('inst_type') == 'SWAP')
        futures = snapshot.filter(pl.col('inst_type') == 'FUTURES')
        
        if swap.is_empty() or futures.is_empty():
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
) -> pl.LazyFrame:
    """
    Build forward curve using Kalman-filtered Nelson-Siegel carry model.
    
    Recipe for store.get_derived(). Use functools.partial to configure:
        recipe = partial(build_forwards_kalman, binning='5m', lambda_ns=0.05)
        lf = store.get_derived(recipe, start, end, cache_name='forwards_kalman_5m')
    """
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data
    df_pillars = prepare_pillars(
        store, inst_family, dates, binning, min_time_to_expiry_hours=min_time_to_expiry_hours
    )
    
    if df_pillars.is_empty():
        return pl.LazyFrame()
    
    # Group by timeMs and prepare snapshots
    unique_times = df_pillars['timeMs'].unique().sort()
    snapshots = []
    
    for timeMs in unique_times:
        snapshot = df_pillars.filter(pl.col('timeMs') == timeMs)
        
        # Separate swap and futures
        swap = snapshot.filter(pl.col('inst_type') == 'SWAP')
        futures = snapshot.filter(pl.col('inst_type') == 'FUTURES')
        
        if swap.is_empty() or futures.is_empty():
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

