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
from okx.recipes.helpers import build_cache_name
from okx.helpers import _get_function_name
from tqdm import tqdm

# =============================================================================
# Helper functions for forward recipes
# =============================================================================

def _find_contiguous_groups(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find contiguous blocks of identical values in a sorted array.
    
    Returns:
        (start_indices, end_indices): Arrays of start/end indices for each group.
    """
    if len(values) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    change_points = np.flatnonzero(np.diff(values)) + 1 if len(values) > 1 else np.array([], dtype=int)
    start_indices = np.concatenate(([0], change_points))
    end_indices = np.concatenate((change_points, [len(values)]))
    return start_indices, end_indices


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
    binning: Optional[str] = None,
    tau_ewma_minutes: float = 5.0,
    min_time_to_expiry_hours: float = 2.0,
    unique_times: Optional[list[int]] = None,
    batch_days: Optional[int] = None,
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
    if verbose:
        start_time = datetime.now()
        strategy = f'{binning} binning' if binning is not None else f'provided timestamps' if unique_times is not None else 'all timestamps'
        print(f"Building PCHIP forwards for {inst_family} with {strategy}")
    
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data with timestamp matching
    df_pillars = prepare_pillars(
        store, inst_family, dates, binning, 
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        drop_pillar_idx=drop_pillar_idx,
        batch_days=batch_days,
        verbose=verbose,
    ).collect()
    
    if verbose:
        time_1 = datetime.now()
        print(f" - Time taken to prepare pillars: {time_1 - start_time}")

    # Convert once to NumPy arrays/lists to avoid per-group DataFrame materialization
    time_values = df_pillars['timeMs'].to_numpy()
    T_values = df_pillars['T'].to_numpy()
    ln_bid_values = df_pillars['ln_bid_1_px'].to_numpy()
    ln_ask_values = df_pillars['ln_ask_1_px'].to_numpy()
    symbol_values = df_pillars['symbol'].to_list()

    # Find contiguous blocks of identical timeMs without creating per-time DataFrames
    start_indices, end_indices = _find_contiguous_groups(time_values)

    curves = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        curve = fit_pchip_curve(
            T_pillars=T_values[start_idx:end_idx],
            F_bid_pillars=ln_bid_values[start_idx:end_idx],
            F_ask_pillars=ln_ask_values[start_idx:end_idx],
            symbols=symbol_values[start_idx:end_idx],
            timeMs=int(time_values[start_idx]),
        )
        curves.append(curve)
    
    if verbose:
        time_2 = datetime.now()
        print(f" - Time taken to fit curves: {time_2 - time_1}")
    # Apply time-aware EWMA smoothing
    smoothed_curves = ewma_smooth(curves, tau_minutes=tau_ewma_minutes)
    if verbose:
        time_3 = datetime.now()
        print(f" - Time taken to smooth curves: {time_3 - time_2}")
    # Convert to Polars
    df_result = curves_to_polars(smoothed_curves)
    if verbose:
        end_time = datetime.now()
        print(f" - Time taken to convert to Polars: {end_time - time_3}")
        print(f" - Total time taken to build PCHIP forwards: {end_time - start_time}")
    return df_result.lazy()


def build_forwards_kalman(
    store,
    dates: list[date],
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = None,
    lambda_ns: float = 1.0,
    tau_minutes: np.ndarray = None,
    sigma_per_sqrt_day: np.ndarray = None,
    min_time_to_expiry_hours: float = 2.0,
    kappa_spread: float = 0.5,
    unique_times: Optional[list[int]] = None,
    drop_pillar_idx: Optional[int] = None,
    batch_days: Optional[int] = None,
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
    if verbose:
        start_time = datetime.now()
        strategy = f'{binning} binning' if binning is not None else f'provided timestamps' if unique_times is not None else 'all timestamps'
        print(f"Building Kalman-filtered Nelson-Siegel forwards for {inst_family} with {strategy}")
    if not dates:
        return pl.LazyFrame()
    
    # Prepare pillar data with timestamp matching
    df_pillars = prepare_pillars(
        store, inst_family, dates, binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        drop_pillar_idx=drop_pillar_idx,
        batch_days=batch_days,
        verbose=verbose,
    ).collect()
    
    if verbose:
        time_1 = datetime.now()
        print(f" - Time taken to prepare and collect pillars: {time_1 - start_time}")

    snapshots = []
    
    for pillars_df in df_pillars.partition_by('timeMs', maintain_order=True):
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
    
    if verbose:
        time_2 = datetime.now()
        print(f" - Time taken to build {len(snapshots)} snapshots: {time_2 - time_1} ")
    
    # Apply time-aware Kalman filter (now expects log prices)
    states = kalman_filter(
        snapshots=snapshots,
        lambda_ns=lambda_ns,
        tau_minutes=tau_minutes,
        sigma_per_sqrt_day=sigma_per_sqrt_day,
        kappa_spread=kappa_spread,
        progress=verbose,
    )
    if verbose:
        time_3 = datetime.now()
        print(f" - Time taken to run Kalman filter: {time_3 - time_2}")

    # Convert to Polars
    df_result = states_to_polars(states)

    if verbose:
        end_time = datetime.now()
        print(f" - Time taken to convert states to Polars: {end_time - time_3}")
        print(f" - Total time taken to build Kalman forwards: {end_time - start_time}")
    return df_result.lazy()

# =============================================================================
# Helper functions for forward assignment
# =============================================================================

def _get_recipe_type(recipe: Callable) -> str:
    """Extract function name from recipe, handles nested functools.partial. via okx.helpers._get_function_name"""
    recipe_name = _get_function_name(recipe).lower()
    if 'pchip' in recipe_name:
        return 'pchip'
    elif 'kalman' in recipe_name:
        return 'kalman'
    else:
        raise ValueError(f"Unknown recipe type: {recipe_name}")

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

def _get_reconstruct_fn(curve_type: str) -> Callable:
    """Get the appropriate reconstruction function and curve type based on recipe name."""
    if curve_type == 'pchip':
        return _reconstruct_forward_pchip
    elif curve_type == 'kalman':
        return _reconstruct_forward_kalman

def _build_curve_lookup(df_forwards: pl.DataFrame, curve_type: str) -> dict[int, object]:
    """Materialize forward curves/states keyed by timestamp for fast lookup."""
    if curve_type == 'pchip':
        from forwards.pchip import PCHIPCurve
        curves = PCHIPCurve.from_polars(df_forwards)
    elif curve_type == 'kalman':
        from forwards.kalman_ns import NSCarryState
        curves = NSCarryState.from_polars(df_forwards)

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
    
    Assumes df_data is already sorted by [timeMs, T] and non-empty.
    
    Steps:
        1. Identify contiguous time groups in sorted data.
        2. For each group, reconstruct forwards for the unique maturities present at that time.
        3. Execute group batches across a thread pool to keep cores busy while staying CPU-bound.
    
    Returns:
        (F_bid_array, F_ask_array, missing_curve_groups)
    """
    n_rows = len(df_data)
    time_values = df_data['timeMs'].to_numpy()
    T_values = df_data['T'].to_numpy()
    start_indices, end_indices = _find_contiguous_groups(time_values)

    F_bid_array = np.full(n_rows, np.nan, dtype=np.float64)
    F_ask_array = np.full(n_rows, np.nan, dtype=np.float64)

    def _process_group_range(range_start: int, range_end: int) -> tuple[int, int]:
        processed = 0
        missing_curve = 0
        for group_idx in range(range_start, range_end):
            processed += 1
            start_idx = start_indices[group_idx]
            end_idx = end_indices[group_idx]

            timeMs = int(time_values[start_idx])
            curve = curves_by_time.get(timeMs)
            if curve is None:
                missing_curve += 1
                continue

            data_T = T_values[start_idx:end_idx]
            unique_T, inverse_indices = np.unique(data_T, return_inverse=True)

            F_bids_unique, F_asks_unique = reconstruct_fn(curve, unique_T)
            F_bids_unique = np.atleast_1d(np.asarray(F_bids_unique, dtype=np.float64))
            F_asks_unique = np.atleast_1d(np.asarray(F_asks_unique, dtype=np.float64))

            F_bid_array[start_idx:end_idx] = F_bids_unique[inverse_indices]
            F_ask_array[start_idx:end_idx] = F_asks_unique[inverse_indices]
        return processed, missing_curve

    # Determine chunk size based on number of groups and number of workers
    group_count = len(start_indices)
    max_workers = min(16, os.cpu_count() or 4)
    chunk_size = max(1, group_count // (max_workers * 8))
    chunk_size = min(chunk_size, 50000)

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

# =============================================================================
# Forward assignment recipe
# =============================================================================

def assign_forwards(
    store,
    lf_data: pl.LazyFrame,
    dates: list[date],
    forwards_recipe: Callable,
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = None,
    batch_days: Optional[int] = None,
    verbose: bool = True,
) -> pl.LazyFrame:
    """
    Assign forward bid/ask prices to a LazyFrame with 'timeMs' and 'T' columns.
    
    This recipe:
    - Fetches forward curves using the provided recipe (via get_derived for caching)
    - Adds F_bid, F_ask columns for each row based on timeMs and T
    
    Recipe for store.get_derived(). Use functools.partial to configure:
        from functools import partial
        forwards_recipe = partial(build_forwards_pchip, tau_ewma_minutes=5.0)
        assignment_recipe = partial(assign_forwards, 
                                    dates=dates, 
                                    forwards_recipe=forwards_recipe,
                                    binning='5m')
        lf_result = assignment_recipe(store, lf_data)
    
    Args:
        store: OrderbookStore instance
        lf_data: LazyFrame with 'timeMs' and 'T' columns
        dates: List of dates to load forward curves for
        forwards_recipe: Pre-configured recipe function (use functools.partial)
        inst_family: Instrument family (e.g., 'BTC-USD')
        binning: Binning interval ('1m', '5m', etc) or None for unbinned
        verbose: Whether to print progress (default: True)
    
    Returns:
        LazyFrame with added columns:
            - F_bid, F_ask (from forward curve)
    """
    curve_type = _get_recipe_type(forwards_recipe)
    if verbose:
        start_time = datetime.now()
        print(f"Assigning {curve_type} forwards to {len(dates)} dates using {curve_type} forwards")
    # Collect the input data (needed for array operations)
    df_data = lf_data.collect()
    if df_data.is_empty():
        return pl.LazyFrame()
    # Validate required columns
    if not {'timeMs', 'T'}.issubset(df_data.columns):
        raise ValueError("LazyFrame must contain columns: {'timeMs', 'T'}")
    if verbose:
        time_1 = datetime.now()
        print(f" - Time taken to collect data: {time_1 - start_time}")

    # =============================================================================
    # Step 1: Configure forwards recipe and fetch forwards dataset
    # =============================================================================

    reconstruct_fn = _get_reconstruct_fn(curve_type)
    forwards_cache_name = build_cache_name(binning, curve_type, forwards_recipe)

    # Configure forwards recipe with appropriate parameters
    if binning is None:
        unique_times = df_data['timeMs'].unique().sort().to_list()
        forwards_recipe = partial(forwards_recipe, unique_times=unique_times, batch_days=batch_days, verbose=verbose)
    else: 
        forwards_recipe = partial(forwards_recipe, binning=binning, batch_days=batch_days, verbose=verbose)

    # Fetch forward dataset
    df_forwards = store.get_derived(
        forwards_recipe,
        dates=dates,
        cache_name=forwards_cache_name,
        verbose=verbose,
    ).collect()
    
    if df_forwards.is_empty():
        return pl.LazyFrame()

    if verbose:
        time_1 = datetime.now()
        print(f" - Time taken to fetch forwards: {time_1 - start_time}")

    # =============================================================================
    # Step 2: Build forward curve lookup
    # =============================================================================

    curves_by_time = _build_curve_lookup(df_forwards, curve_type)
    if verbose:
        time_2 = datetime.now()
        print(f" - Time taken to build forward lookup: {time_2 - time_1}")

    # =============================================================================
    # Step 3: Match data with forwards
    # =============================================================================

    df_data = df_data.sort('timeMs')
    F_bid_array, F_ask_array, missing_curve_groups = _populate_forward_arrays(
        df_data,
        curves_by_time,
        reconstruct_fn,
        verbose,
    )

    df_data = df_data.with_columns([
        pl.Series('F_bid', F_bid_array),
        pl.Series('F_ask', F_ask_array),
    ])

    if verbose:
        end_time = datetime.now()
        matched_mask = np.isfinite(F_bid_array) & np.isfinite(F_ask_array)
        matched_count = int(matched_mask.sum())
        print(f" - Time taken to match forwards: {end_time - time_2}, matched {matched_count:,} / {len(df_data):,} rows")
        print(f" - Total time taken to assign forwards: {end_time - start_time}")
        if missing_curve_groups:
            print(f"Skipped {missing_curve_groups:,} timestamp groups with no forward curve")

    return df_data.lazy()