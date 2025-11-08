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
from typing import Optional, Tuple

from forwards.pchip import fit_pchip_curve, ewma_smooth, curves_to_polars, PCHIPCurve
from forwards.kalman_ns import kalman_filter, states_to_polars
from okx.recipes.helpers import early_roll, finalize_binning

# =============================================================================
# Pillar preparation
# =============================================================================


def prepare_pillars(
    store,
    inst_family: str,
    dates: list[date],
    binning: Optional[str] = None,
    min_time_to_expiry_hours: float = 2.0,
    unique_times: Optional[list[int]] = None,
    drop_pillar_idx: Optional[int] = None,
) -> dict[int, pl.DataFrame]:
    import time
    start_time = time.time()
    """
    Prepare concatenated pillar data from SWAP and FUTURES orderbooks.
    
    Returns dict mapping timeMs -> pillars_df where each DataFrame contains
    SWAP (at T=0) followed by FUTURES sorted by maturity. All data is in log-space.
    """
    # =============================================================================
    # Build feature list (ordered for performance: strip early to reduce columns)
    # =============================================================================
    features_base = ['trim', 'strip']
    if binning:
        features_base.append('bin')
        features_base.append(finalize_binning)
    features_base.extend(['spread', 'rel_spread', 'tenor', 'log'])  # Added 'log' feature
    
    # SWAP features (no early-roll filter)
    features_swap = features_base.copy()
    
    # FUTURES features (add early-roll filter after tenor)
    features_futures = features_base.copy()
    features_futures.append(early_roll(min_time_to_expiry_hours))
    
    # =============================================================================
    # Fetch SWAP and FUTURES data
    # =============================================================================

    cache_name = 'full_pillars' if binning is None else f'{binning}_pillars'
    
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
    
    get_time = time.time()
    print(f"get took {get_time - start_time:.3f} seconds")
    
    
    # Collect
    df_swap = lf_swap.collect()
    df_futures = lf_futures.collect()

    collect_time = time.time()
    print(f"collect took {collect_time - get_time:.3f} seconds")

    if df_swap.is_empty() or df_futures.is_empty():
        return {}
    
    # =============================================================================
    # Concatenation and dict building
    # =============================================================================
    if unique_times is not None:
        times = sorted(unique_times)
    else:
        # Union of all timestamps from both dataframes
        times = pl.concat([
            df_swap.select('timeMs'),
            df_futures.select('timeMs')
        ]).unique().sort('timeMs')['timeMs'].to_list()
    
    concat_time = time.time()
    print(f"concat took {concat_time - collect_time:.3f} seconds")
    
    # Build snapshot dict (timeMs -> pillars_df)
    snapshots = {}
    df_swap_sorted = df_swap.sort('timeMs')
    unique_symbols = df_futures['symbol'].unique().to_list()
    
    for time in times:
        df_time = pl.DataFrame({'timeMs': [time]})
        
        # Get most recent swap snapshot at or before this time
        swap_matched = df_time.join_asof(df_swap_sorted, on='timeMs', strategy='backward')
        if swap_matched.is_empty() or swap_matched['ln_bid_1_px'][0] is None:
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
                (pl.col('ln_bid_1_px').is_not_null())
            )
            if not matched.is_empty():
                futures_at_time.append(matched)
        
        if not futures_at_time:
            continue
        
        # Concatenate all valid futures for this time and sort by T
        futures_df = pl.concat(futures_at_time).sort('T')
        
        # Concatenate swap (T=0) with futures to create full pillars DataFrame
        pillars_df = pl.concat([swap_matched, futures_df]).sort('T')
        
        # Drop pillar if specified (for LOEO evaluation)
        if drop_pillar_idx is not None:
            if drop_pillar_idx < len(pillars_df):
                # Drop the specified pillar by index
                pillars_df = pl.concat([
                    pillars_df[:drop_pillar_idx],
                    pillars_df[drop_pillar_idx + 1:]
                ])
            else:
                # Index out of range, skip this snapshot
                continue
        
        snapshots[time] = pillars_df
    
    return snapshots

# =============================================================================
# Helper functions for forward recipes
# =============================================================================

def _extract_pillar_arrays(pillars_df: pl.DataFrame) -> dict:
    """Extract arrays from concatenated pillars DataFrame (already sorted by T)."""
    return {
        'timeMs': int(pillars_df['timeMs'][0]),
        'T': pillars_df['T'].to_numpy(),
        'ln_F_bid': pillars_df['ln_bid_1_px'].to_numpy(),
        'ln_F_ask': pillars_df['ln_ask_1_px'].to_numpy(),
        'rel_spreads': pillars_df['rel_spread'].to_numpy(),
        'symbols': pillars_df['symbol'].to_list(),
    }

# =============================================================================
# Forward curve building recipes
# =============================================================================

def build_forwards_pchip(
    store,
    dates: list[date],
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = '5m',
    tau_ewma_minutes: float = 5.0,
    min_time_to_expiry_hours: float = 2.0,
    unique_times: Optional[list[int]] = None,
    drop_pillar_idx: Optional[int] = None,
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
        drop_pillar_idx: Optional pillar index to exclude (for LOEO evaluation).
                        If provided, the pillar at this index in the sorted
                        futures DataFrame will be dropped before fitting.
    """
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data with timestamp matching
    snapshots = prepare_pillars(
        store, inst_family, dates, binning, 
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        drop_pillar_idx=drop_pillar_idx,
    )
    
    if not snapshots:
        return pl.LazyFrame()
    
    curves = []
    
    for timeMs, pillars_df in snapshots.items():
        # Extract arrays from concatenated pillars
        data = _extract_pillar_arrays(pillars_df)
        
        # Fit PCHIP curve (data already has log prices from prepare_pillars)
        curve = fit_pchip_curve(
            T_pillars=data['T'],
            F_bid_pillars=data['ln_F_bid'],
            F_ask_pillars=data['ln_F_ask'],
            symbols=data['symbols'],
            timeMs=int(timeMs),
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
    drop_pillar_idx: Optional[int] = None,
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
        drop_pillar_idx: Optional pillar index to exclude (for LOEO evaluation).
                        If provided, the pillar at this index in the sorted
                        futures DataFrame will be dropped before fitting.
    """
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data with timestamp matching
    snapshots_dict = prepare_pillars(
        store, inst_family, dates, binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        drop_pillar_idx=drop_pillar_idx,
    )
    
    if not snapshots_dict:
        return pl.LazyFrame()
    
    snapshots = []
    
    for timeMs, pillars_df in snapshots_dict.items():
        # Extract arrays from concatenated pillars
        data = _extract_pillar_arrays(pillars_df)
        
        # Create snapshot dict for kalman_filter
        snapshot = {
            'timeMs': data['timeMs'],
            'T': data['T'],
            'ln_F_bid': data['ln_F_bid'],
            'ln_F_ask': data['ln_F_ask'],
            'rel_spreads': data['rel_spreads'],
        }
        snapshots.append(snapshot)
    
    if not snapshots:
        return pl.LazyFrame()
    
    # Apply time-aware Kalman filter (now expects log prices)
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