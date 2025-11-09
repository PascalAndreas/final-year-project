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
    cache_name_suffix: str = "pillars",
) -> pl.LazyFrame:
    """
    Prepare concatenated pillar data from SWAP and FUTURES orderbooks.

    Returns a LazyFrame sorted by (timeMs, T) that contains the latest snapshot
    for each instrument at or before every requested timestamp. This keeps the
    function compatible with store.get_derived() caching and avoids Python-side
    looping over symbols/timestamps.
    """
    if not dates:
        return pl.DataFrame().lazy()

    # Build feature list (ordered for performance: strip early to reduce columns)
    features_base = ['trim', 'strip']
    if binning:
        features_base.extend(['bin', finalize_binning])
    features_base.extend(['spread', 'rel_spread', 'tenor', 'log'])

    features_swap = features_base.copy()

    features_futures = features_base.copy()
    features_futures.append(early_roll(min_time_to_expiry_hours))

    suffix = cache_name_suffix or "pillars"
    prefix = "full" if binning is None else binning
    cache_name = f"{prefix}_{suffix}"

    shared_params = {
        'inst_family': inst_family,
        'dates': dates,
        'depth': 1,
        'binning': binning,
        'cache_name': cache_name,
    }

    lf_swap = store.get(
        inst_type='SWAP',
        features=features_swap,
        **shared_params,
    ).sort('timeMs')

    lf_futures = store.get(
        inst_type='FUTURES',
        features=features_futures,
        **shared_params,
    ).sort(['symbol', 'timeMs'])

    if unique_times is not None:
        lf_times = pl.DataFrame({'timeMs': sorted(unique_times)}).lazy()
    else:
        lf_times = (
            pl.concat([
                lf_swap.select('timeMs'),
                lf_futures.select('timeMs'),
            ])
            .unique()
            .sort('timeMs')
        )

    # Align swap snapshots to requested timestamps (latest observation <= timeMs)
    lf_swap_snapshots = (
        lf_times.join_asof(
            lf_swap,
            on='timeMs',
            strategy='backward',
            suffix='_hist',
        )
        .drop('timeMs_hist', strict=False)
        .filter(pl.col('ln_bid_1_px').is_not_null())
    )

    lf_valid_times = (
        lf_swap_snapshots.select('timeMs')
        .unique()
        .sort('timeMs')
    )

    # Align each futures symbol independently using cross join + asof
    lf_symbol_times = (
        lf_futures.select('symbol')
        .unique()
        .join(lf_valid_times, how='cross')
        .sort(['symbol', 'timeMs'])
    )
    lf_futures_snapshots = (
        lf_symbol_times.join_asof(
            lf_futures,
            on='timeMs',
            by='symbol',
            strategy='backward',
            suffix='_hist',
        )
        .drop(['symbol_hist', 'timeMs_hist'], strict=False)
        .filter(pl.col('ln_bid_1_px').is_not_null())
        .filter(pl.col('expiry') > pl.col('timeMs'))
    )

    # Ensure consistent column order before concat (LazyFrames don't have .columns)
    common_cols = ['timeMs', 'symbol', 'rel_spread', 'expiry', 'T', 
                   'ln_bid_1_px', 'ln_ask_1_px']

    pillars = pl.concat([
        lf_swap_snapshots.select(common_cols),
        lf_futures_snapshots.select(common_cols),
    ])

    pillars = (
        pillars
        .sort(['timeMs', 'T'])
        .with_columns([
            pl.int_range(pl.len()).over('timeMs').alias('pillar_idx'),
            pl.len().over('timeMs').alias('_pillars_per_time'),
        ])
    )

    if drop_pillar_idx is not None:
        pillars = pillars.filter(
            pl.col('_pillars_per_time') > drop_pillar_idx
        ).filter(pl.col('pillar_idx') != drop_pillar_idx)

    return (
        pillars
        .drop('_pillars_per_time')
        .sort(['timeMs', 'T'])
    )

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
    start_time = datetime.now()
    if not dates:
        return pl.DataFrame().lazy()
    
    # Prepare pillar data with timestamp matching
    pillars_lf = prepare_pillars(
        store, inst_family, dates, binning, 
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        drop_pillar_idx=drop_pillar_idx,
    )
    
    df_pillars = pillars_lf.collect()
    
    if df_pillars.is_empty():
        return pl.DataFrame().lazy()
    
    time_1 = datetime.now()
    print(f"Time taken to prepare pillars: {time_1 - start_time}")

    # Convert once to NumPy arrays/lists to avoid per-group DataFrame materialization
    df_pillars = df_pillars.sort(['timeMs', 'T'])
    time_values = df_pillars['timeMs'].to_numpy()
    T_values = df_pillars['T'].to_numpy()
    ln_bid_values = df_pillars['ln_bid_1_px'].to_numpy()
    ln_ask_values = df_pillars['ln_ask_1_px'].to_numpy()
    symbol_values = df_pillars['symbol'].to_list()

    if len(time_values) == 0:
        return pl.DataFrame().lazy()

    # Find contiguous blocks of identical timeMs without creating per-time DataFrames
    change_points = np.flatnonzero(np.diff(time_values)) + 1 if len(time_values) > 1 else np.array([], dtype=int)
    start_indices = np.concatenate(([0], change_points))
    end_indices = np.concatenate((change_points, [len(time_values)]))

    curves = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        if end_idx - start_idx < 2:
            continue

        curve = fit_pchip_curve(
            T_pillars=T_values[start_idx:end_idx],
            F_bid_pillars=ln_bid_values[start_idx:end_idx],
            F_ask_pillars=ln_ask_values[start_idx:end_idx],
            symbols=symbol_values[start_idx:end_idx],
            timeMs=int(time_values[start_idx]),
        )
        curves.append(curve)
    
    if not curves:
        return pl.DataFrame().lazy()
    
    time_2 = datetime.now()
    print(f"Time taken to fit curves: {time_2 - time_1}")
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
        return pl.DataFrame().lazy()
    
    # Prepare pillar data with timestamp matching
    pillars_lf = prepare_pillars(
        store, inst_family, dates, binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        drop_pillar_idx=drop_pillar_idx,
    )
    
    df_pillars = pillars_lf.collect()
    
    if df_pillars.is_empty():
        return pl.DataFrame().lazy()
    
    snapshots = []
    
    for pillars_df in df_pillars.partition_by('timeMs', maintain_order=True):
        # Extract arrays from concatenated pillars
        data = _extract_pillar_arrays(pillars_df)
        if len(data['T']) < 2:
            continue
        
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
        return pl.DataFrame().lazy()
    
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
