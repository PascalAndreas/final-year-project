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
from forwards.utils import compute_weights


def _drop_unneeded_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Drop columns not needed for forward curve fitting."""
    needed_columns = ['timeMs', 'symbol', 'bid_1_px', 'ask_1_px']
    # Keep only columns that exist in the frame
    existing_columns = lf.collect_schema().names()
    keep_columns = [col for col in needed_columns if col in existing_columns]
    return lf.select(keep_columns)


def _finalize_binning(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Finalize binning by promoting time_bin to timeMs and dropping old time columns.
    
    After binning, time_bin contains the aligned timestamps as datetime.
    This function converts it to epoch milliseconds (int64) to match timeMs format.
    """
    return (lf
        .drop('timeMs')
        .with_columns(pl.col('time_bin').dt.epoch('ms').alias('timeMs'))
        .drop('time_bin')
    )


def _make_early_roll_filter(min_time_to_expiry_hours: float):
    """Create early-roll filter callable for use in feature pipeline."""
    def filter_fn(lf: pl.LazyFrame) -> pl.LazyFrame:
        min_seconds = min_time_to_expiry_hours * 3600
        time_to_expiry_seconds = (pl.col('expiry') - pl.col('timeMs')) / 1000.0
        return lf.filter(time_to_expiry_seconds >= min_seconds)
    return filter_fn


def prepare_pillars(
    store,
    inst_family: str,
    dates: list[date],
    binning: Optional[str] = None,
    min_time_to_expiry_hours: float = 2.0,
    cache_name_suffix: str = "_pillars",
    unique_times: Optional[list[int]] = None,
) -> dict[int, tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Prepare pillar data from SWAP and FUTURES orderbooks with timestamp matching.
    
    Returns dict mapping timeMs -> (swap_row, futures_df) where each entry contains
    the most recent swap snapshot and all valid futures contracts at that time.
    """
    # Build feature list (ordered for performance: trim early to reduce columns)
    features_base = ['trim', _drop_unneeded_columns]
    if binning:
        features_base.append('bin')
    features_base.extend(['spread', 'rel_spread', 'tenor'])
    if binning:
        features_base.append(_finalize_binning)
    
    # SWAP features (no early-roll filter)
    features_swap = features_base.copy()
    
    # FUTURES features (add early-roll filter after tenor)
    features_futures = features_base.copy()
    features_futures.append(_make_early_roll_filter(min_time_to_expiry_hours))
    
    # Build cache name if binning
    cache_name = f"{binning}{cache_name_suffix}" if binning else None
    
    # Shared parameters
    shared_params = {
        'inst_family': inst_family,
        'dates': dates,
        'depth': 1,
        'binning': binning,
        'cache_name': cache_name,
    }
    
    # Fetch SWAP (perpetual)
    lf_swap = store.get(
        inst_type='SWAP',
        features=features_swap,
        **shared_params,
    )
    
    # Fetch FUTURES
    lf_futures = store.get(
        inst_type='FUTURES',
        features=features_futures,
        **shared_params,
    )
    
    # Collect
    df_swap = lf_swap.collect()
    df_futures = lf_futures.collect()
    
    if df_swap.is_empty() or df_futures.is_empty():
        return {}
    
    # Determine timestamps
    if unique_times is not None:
        times = sorted(unique_times)
    else:
        # Union of all timestamps from both dataframes
        times = pl.concat([
            df_swap.select('timeMs'),
            df_futures.select('timeMs')
        ]).unique().sort('timeMs')['timeMs'].to_list()
    
    # Build snapshot dict
    snapshots = {}
    df_swap_sorted = df_swap.sort('timeMs')
    unique_symbols = df_futures['symbol'].unique().to_list()
    
    for time in times:
        df_time = pl.DataFrame({'timeMs': [time]})
        
        # Get most recent swap snapshot at or before this time
        swap_matched = df_time.join_asof(df_swap_sorted, on='timeMs', strategy='backward')
        if swap_matched.is_empty() or swap_matched['bid_1_px'][0] is None:
            continue
        
        # Get most recent snapshot for each futures symbol at or before this time
        futures_at_time = []
        for symbol in unique_symbols:
            symbol_data = df_futures.filter(pl.col('symbol') == symbol).sort('timeMs')
            if symbol_data.is_empty():
                continue
            matched = df_time.join_asof(symbol_data, on='timeMs', strategy='backward')
            # Filter where expiry > timeMs (not expired) and has valid data
            matched = matched.filter(
                (pl.col('expiry') > pl.col('timeMs')) & 
                (pl.col('bid_1_px').is_not_null())
            )
            if not matched.is_empty():
                futures_at_time.append(matched)
        
        if not futures_at_time:
            continue
        
        # Concatenate all valid futures for this time
        futures_df = pl.concat(futures_at_time)
        snapshots[time] = (swap_matched, futures_df)
    
    return snapshots


def build_forwards_pchip(
    store,
    dates: list[date],
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = '5m',
    tau_ewma_minutes: float = 5.0,
    w0_anchor: float = 10.0,
    min_time_to_expiry_hours: float = 2.0,
    max_weight_ratio: float = 100.0,
    unique_times: Optional[list[int]] = None,
) -> pl.LazyFrame:
    """
    Build forward curve using PCHIP interpolation with time-aware EWMA smoothing.
    
    Uses α(Δt) = exp(-Δt/τ) for frame-rate invariance across binning intervals.
    
    Recipe for store.get_derived(). Use functools.partial to configure:
        recipe = partial(build_forwards_pchip, binning='1m', tau_ewma_minutes=5.0)
        lf = store.get_derived(recipe, start, end, cache_name='forwards_pchip_1m')
    
    Args:
        tau_ewma_minutes: Time constant in minutes (typical: 3-10 for crypto)
                         Half-life = tau * ln(2) ≈ 0.693 * tau
        unique_times: Optional list of timeMs values to build curves for.
                     If None and binning is set, uses binned timestamps.
                     If None and no binning, uses union of all timestamps.
    """
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data with timestamp matching
    snapshots = prepare_pillars(
        store, inst_family, dates, binning, 
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times
    )
    
    if not snapshots:
        return pl.LazyFrame()
    
    curves = []
    
    for timeMs, (swap, futures) in snapshots.items():
        
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
    
    # Apply time-aware EWMA smoothing
    smoothed_curves = ewma_smooth(curves, tau_minutes=tau_ewma_minutes)
    
    # Convert to Polars
    df_result = curves_to_polars(smoothed_curves)
    
    return df_result.lazy()


def build_forwards_kalman(
    store,
    dates: list[date],
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = '1m',
    lambda_ns: float = 1.0,
    tau_minutes: np.ndarray = None,
    sigma_per_sqrt_day: np.ndarray = None,
    min_time_to_expiry_hours: float = 2.0,
    kappa_spread: float = 0.5,
    unique_times: Optional[list[int]] = None,
) -> pl.LazyFrame:
    """
    Build forward curve using time-aware Kalman-filtered Nelson-Siegel carry model.
    
    Uses exact OU discretization for frame-rate invariance and spread-based
    measurement noise for adaptive filtering.
    
    Recipe for store.get_derived(). Use functools.partial to configure:
        recipe = partial(build_forwards_kalman, binning='5m', lambda_ns=1.0)
        lf = store.get_derived(recipe, start, end, cache_name='forwards_kalman_5m')
    
    Args:
        lambda_ns: Shape parameter (0.5-2.0/year typical for crypto)
        tau_minutes: Time constants [τ0, τ1, τ2] in minutes (default: [2d, 5d, 10d])
        sigma_per_sqrt_day: Volatilities [σ0, σ1, σ2] per sqrt(day) (default: [0.01, 0.01, 0.01])
        kappa_spread: Scale factor for spread-based measurement noise (0.5-1.0 typical)
        unique_times: Optional list of timeMs values to build curves for.
                     If None and binning is set, uses binned timestamps.
                     If None and no binning, uses union of all timestamps.
    """
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data with timestamp matching
    snapshots_dict = prepare_pillars(
        store, inst_family, dates, binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times
    )
    
    if not snapshots_dict:
        return pl.LazyFrame()
    
    snapshots = []
    
    for timeMs, (swap, futures) in snapshots_dict.items():
        
        # Extract swap reference
        F_ref_bid = swap['bid_1_px'][0]
        F_ref_ask = swap['ask_1_px'][0]
        
        # Extract futures pillars
        T_pillars = futures['T'].to_numpy()
        F_bid_pillars = futures['bid_1_px'].to_numpy()
        F_ask_pillars = futures['ask_1_px'].to_numpy()
        rel_spreads = futures['rel_spread'].to_numpy()
        
        snapshots.append({
            'timeMs': int(timeMs),
            'T_pillars': T_pillars,
            'F_bid_pillars': F_bid_pillars,
            'F_ask_pillars': F_ask_pillars,
            'rel_spreads': rel_spreads,
            'F_ref_bid': F_ref_bid,
            'F_ref_ask': F_ref_ask,
        })
    
    if not snapshots:
        return pl.LazyFrame()
    
    # Apply time-aware Kalman filter
    states = kalman_filter(
        snapshots=snapshots,
        lambda_ns=lambda_ns,
        tau_minutes=tau_minutes,
        sigma_per_sqrt_day=sigma_per_sqrt_day,
        kappa_spread=kappa_spread,
    )
    
    # Convert to Polars
    df_result = states_to_polars(states)
    
    return df_result.lazy()

