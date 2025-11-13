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

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import numpy as np
from datetime import datetime, date
from typing import Callable, Optional
from functools import partial

from forwards.pchip import fit_pchip_curve, ewma_smooth, curves_to_polars
from forwards.kalman_ns import kalman_filter, states_to_polars
from okx.recipes.pillars import prepare_pillars
from tqdm import tqdm

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
    verbose: bool = False,
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
        batch_days=3,
    )
    
    df_pillars = pillars_lf.collect()
    
    if df_pillars.is_empty():
        return pl.DataFrame().lazy()
    
    time_1 = datetime.now()
    if verbose:
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
    if verbose:
        print(f"Time taken to fit curves: {time_2 - time_1}")
    # Apply time-aware EWMA smoothing
    smoothed_curves = ewma_smooth(curves, tau_minutes=tau_ewma_minutes)
    time_3 = datetime.now()
    if verbose:
        print(f"Time taken to smooth curves: {time_3 - time_2}")
    # Convert to Polars
    df_result = curves_to_polars(smoothed_curves)
    time_4 = datetime.now()
    if verbose:
        print(f"Time taken to convert to Polars: {time_4 - time_3}")
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
    verbose: bool = False,
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
    start_time = datetime.now()
    if not dates:
        return pl.DataFrame().lazy()
    
    # Prepare pillar data with timestamp matching
    pillars_lf = prepare_pillars(
        store, inst_family, dates, binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        drop_pillar_idx=drop_pillar_idx,
        batch_days=3,
    )
    
    df_pillars = pillars_lf.collect()
    
    if df_pillars.is_empty():
        return pl.DataFrame().lazy()
    
    time_1 = datetime.now()
    if verbose:
        print(f"Time taken to prepare pillars: {time_1 - start_time}")

    snapshots = []
    snapshot_start = datetime.now()
    
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
    
    snapshot_end = datetime.now()
    if verbose:
        print(
            f"Time taken to build snapshots: {snapshot_end - snapshot_start} "
            f"({len(snapshots):,} usable timestamps)"
        )

    if not snapshots:
        return pl.DataFrame().lazy()
    
    filter_start = datetime.now()
    # Apply time-aware Kalman filter (now expects log prices)
    states = kalman_filter(
        snapshots=snapshots,
        lambda_ns=lambda_ns,
        tau_minutes=tau_minutes,
        sigma_per_sqrt_day=sigma_per_sqrt_day,
        kappa_spread=kappa_spread,
        progress=verbose,
    )
    filter_end = datetime.now()
    if verbose:
        print(f"Time taken to run Kalman filter: {filter_end - filter_start}")

    convert_start = datetime.now()
    # Convert to Polars
    df_result = states_to_polars(states)
    convert_end = datetime.now()
    if verbose:
        print(f"Time taken to convert states: {convert_end - convert_start}")

    return df_result.lazy()


# =============================================================================
# Helper functions for forward assignment
# =============================================================================

def _get_recipe_name(recipe: Callable) -> str:
    """Extract function name from recipe (handles functools.partial)."""
    if isinstance(recipe, partial):
        return recipe.func.__name__
    return recipe.__name__


def _reconstruct_forward_pchip(curve, T_target: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Reconstruct forward from a PCHIPCurve object."""
    from forwards.pchip import reconstruct_forward
    return reconstruct_forward(curve, T_target)


def _reconstruct_forward_kalman(state, T_target: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Reconstruct forward from Kalman NS state."""
    from forwards.kalman_ns import reconstruct_ns_forward
    F_bid = reconstruct_ns_forward(state, T_target, use_bid=True)
    F_ask = reconstruct_ns_forward(state, T_target, use_bid=False)
    return F_bid, F_ask


def _get_reconstruct_fn(forwards_recipe: Callable) -> tuple[str, Callable]:
    """Get the appropriate reconstruction function and curve type based on recipe name."""
    recipe_name = _get_recipe_name(forwards_recipe)
    if 'pchip' in recipe_name.lower():
        return 'pchip', _reconstruct_forward_pchip
    elif 'kalman' in recipe_name.lower():
        return 'kalman', _reconstruct_forward_kalman
    else:
        raise ValueError(f"Unknown recipe type: {recipe_name}")


def _build_curve_lookup(df_forwards: pl.DataFrame, curve_type: str) -> dict[int, object]:
    """Materialize forward curves/states keyed by timestamp for fast lookup."""
    if curve_type == 'pchip':
        from forwards.pchip import PCHIPCurve
        curves = PCHIPCurve.from_polars(df_forwards)
    elif curve_type == 'kalman':
        from forwards.kalman_ns import NSCarryState
        curves = NSCarryState.from_polars(df_forwards)
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}")

    if isinstance(curves, list):
        curve_list = curves
    else:
        curve_list = [curves]

    return {curve.timeMs: curve for curve in curve_list}


def _populate_forward_arrays(
    df_data: pl.DataFrame,
    curves_by_time: dict[int, object],
    reconstruct_fn: Callable,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Assign forward bids/asks to every timestamp in parallel.
    
    Steps:
        1. Sort data by timestamp and identify contiguous groups.
        2. For each group, reconstruct forwards for the unique maturities present at that time.
        3. Execute group batches across a thread pool to keep cores busy while staying CPU-bound.
    
    Returns:
        (F_bid_array, F_ask_array, missing_curve_groups)
    """
    n_rows = len(df_data)
    if n_rows == 0:
        return np.array([]), np.array([]), 0

    time_values = df_data['timeMs'].to_numpy()
    T_values = df_data['T'].to_numpy()
    sort_idx = np.argsort(time_values, kind='mergesort')
    sorted_times = time_values[sort_idx]
    change_points = np.flatnonzero(np.diff(sorted_times)) + 1 if len(sorted_times) > 1 else np.array([], dtype=int)
    start_indices = np.concatenate(([0], change_points))
    end_indices = np.concatenate((change_points, [len(sorted_times)]))

    F_bid_array = np.full(n_rows, np.nan, dtype=np.float64)
    F_ask_array = np.full(n_rows, np.nan, dtype=np.float64)
    group_count = len(start_indices)
    if group_count == 0:
        return F_bid_array, F_ask_array, 0

    def _process_group_range(range_start: int, range_end: int) -> tuple[int, int]:
        processed = 0
        missing_curve = 0
        for group_idx in range(range_start, range_end):
            processed += 1
            start_idx_group = start_indices[group_idx]
            end_idx_group = end_indices[group_idx]
            row_indices = sort_idx[start_idx_group:end_idx_group]
            if len(row_indices) == 0:
                continue

            timeMs = int(sorted_times[start_idx_group])
            curve = curves_by_time.get(timeMs)
            if curve is None:
                missing_curve += 1
                continue

            data_T = T_values[row_indices]
            if data_T.size == 0:
                continue

            unique_T, inverse_indices = np.unique(data_T, return_inverse=True)
            if unique_T.size == 0:
                continue

            F_bids_unique, F_asks_unique = reconstruct_fn(curve, unique_T)
            F_bids_unique = np.atleast_1d(np.asarray(F_bids_unique, dtype=np.float64))
            F_asks_unique = np.atleast_1d(np.asarray(F_asks_unique, dtype=np.float64))

            F_bid_array[row_indices] = F_bids_unique[inverse_indices]
            F_ask_array[row_indices] = F_asks_unique[inverse_indices]
        return processed, missing_curve

    max_workers = min(32, max(1, (os.cpu_count() or 4) - 1))
    chunk_size = max(1, group_count // (max_workers * 8) or 1)
    chunk_size = min(chunk_size, 50_000)

    progress = tqdm(total=group_count, desc="Matching forwards") if verbose else None
    missing_curve_groups = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunk_start in range(0, group_count, chunk_size):
            chunk_end = min(group_count, chunk_start + chunk_size)
            futures.append(executor.submit(_process_group_range, chunk_start, chunk_end))
        for future in as_completed(futures):
            processed, missing = future.result()
            missing_curve_groups += missing
            if progress:
                progress.update(processed)
    if progress:
        progress.close()

    return F_bid_array, F_ask_array, missing_curve_groups


def _format_cache_value(value) -> str:
    """Format parameter value for cache name (e.g., 5.0 -> '5.0', '5m' -> '5m')."""
    if isinstance(value, float):
        # Format floats to remove trailing zeros
        return f"{value:.10g}"
    elif isinstance(value, (int, str)):
        return str(value)
    elif value is None:
        return "None"
    else:
        # For other types, use simple string representation
        return str(value).replace(" ", "")


def _build_cache_name(curve_type: str, forwards_recipe: Callable, binning: Optional[str] = None) -> str:
    """Build cache name like '{binning|full}_{curve}_{params}' for forwards recipe."""
    if isinstance(forwards_recipe, partial):
        params = forwards_recipe.keywords or {}
        param_parts = [
            f"{key}={_format_cache_value(params[key])}"
            for key in sorted(params)
        ]
        params_str = "_".join(param_parts)
    else:
        params_str = ""
    prefix = binning if binning is not None else "full"
    recipe_part = curve_type
    if params_str:
        recipe_part = f"{recipe_part}_{params_str}"
    return f"{prefix}_{recipe_part}_forwards"

# =============================================================================
# Forward assignment recipe
# =============================================================================

def assign_forwards(
    store,
    df_data: pl.DataFrame,
    dates: list[date],
    forwards_recipe: Callable,
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Assign forward bid/ask prices to a DataFrame with 'timeMs' and 'T' columns.
    
    This function:
    - Fetches forward curves using the provided recipe (via get_derived for caching)
    - Adds F_bid, F_ask columns for each row based on timeMs and T
    
    Args:
        df_data: DataFrame with 'timeMs' and 'T' columns
        store: OrderbookStore instance
        dates: List of dates to load forward curves for
        forwards_recipe: Pre-configured recipe function (use functools.partial)
        inst_family: Instrument family (e.g., 'BTC-USD')
        binning: Binning interval ('1m', '5m', etc) or None for unbinned
        verbose: Whether to print progress (default: True)
    
    Returns:
        DataFrame with added columns:
            - F_bid, F_ask (from forward curve)
    """
    if df_data.is_empty():
        return pl.DataFrame()
    
    # Validate required columns
    required_cols = {'timeMs', 'T'}
    if not required_cols.issubset(df_data.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    start_time = datetime.now()
    
    # =============================================================================
    # Step 1: Configure forwards recipe and fetch forwards dataset
    # =============================================================================
    curve_type, reconstruct_fn = _get_reconstruct_fn(forwards_recipe)
    forwards_cache_name = _build_cache_name(curve_type, forwards_recipe, binning)

    # Configure forwards recipe with appropriate parameters
    if binning is None:
        unique_times = df_data['timeMs'].unique().sort().to_list()
        forwards_recipe = partial(forwards_recipe, unique_times=unique_times)
    else: 
        forwards_recipe = partial(forwards_recipe, binning=binning)

    # Fetch forward dataset
    lf_forwards = store.get_derived(
        forwards_recipe,
        dates=dates,
        cache_name=forwards_cache_name,
        verbose=verbose,
    )
    df_forwards = lf_forwards.collect()
    
    if df_forwards.is_empty():
        return pl.DataFrame()

    if verbose:
        time_1 = datetime.now()
        print(f"Time taken to fetch forwards: {time_1 - start_time}")

    # =============================================================================
    # Step 2: Build forward curve lookup
    # =============================================================================
    curves_by_time = _build_curve_lookup(df_forwards, curve_type)
    if not curves_by_time:
        return pl.DataFrame()
    if verbose:
        time_2 = datetime.now()
        print(f"Time taken to build forward lookup: {time_2 - time_1}")

    # =============================================================================
    # Step 3: Match data with forwards
    # =============================================================================
    match_start = datetime.now()
    F_bid_array, F_ask_array, missing_curve_groups = _populate_forward_arrays(
        df_data,
        curves_by_time,
        reconstruct_fn,
        verbose,
    )
    if F_bid_array.size == 0:
        return pl.DataFrame()

    df_data = df_data.with_columns([
        pl.Series('F_bid', F_bid_array),
        pl.Series('F_ask', F_ask_array),
    ])

    if verbose:
        time_3 = datetime.now()
        matched_mask = np.isfinite(F_bid_array) & np.isfinite(F_ask_array)
        matched_count = int(matched_mask.sum())
        print(
            f"Time taken to match forwards: core {time_3 - match_start}, "
            f"incl. setup {time_3 - time_2} (matched {matched_count:,} / {len(df_data):,} rows)"
        )
        if missing_curve_groups:
            print(f"Skipped {missing_curve_groups:,} timestamp groups with no forward curve")

    return df_data