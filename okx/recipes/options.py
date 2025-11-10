"""
Options-based evaluation for forward curves.

Compare fitted forward curves to options-implied forwards via put-call parity.
This provides an independent validation that forward curves are consistent with
market-implied information from option prices.

Price Units Convention:
-----------------------
All prices in the OKX data are in USD for BTC-USD instruments:
- Options (calls/puts): Premium in USD
- SWAP: Price in USD  
- Forwards: Price in USD

Put-Call Parity:
    C - P = (F - K) for zero interest rate (crypto convention)
    F_implied = K + (C - P)

Note: Strike prices are also in USD.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import numpy as np
from datetime import date, datetime
from typing import Callable, Optional
from functools import partial

from okx.recipes.helpers import early_roll, finalize_binning
from tqdm import tqdm

# =============================================================================
# Helper functions
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


def _fetch_options_dataset(
    store,
    inst_family: str,
    dates: list[date],
    binning: Optional[str],
    min_time_to_expiry_hours: float,
    verbose: bool,
) -> tuple[pl.DataFrame, list[int]]:
    start_time = datetime.now()
    cache_name = 'full_options' if binning is None else f'{binning}_options'

    options_features = ['trim', 'strip']
    if binning:
        options_features.extend(['bin', finalize_binning])
    else:
        options_features.extend(['dedupe'])
    options_features.extend(['tenor', early_roll(min_time_to_expiry_hours), 'parse_option'])

    df_options = store.get(
        inst_type='OPTION',
        inst_family=inst_family,
        dates=dates,
        depth=1,
        binning=binning,
        features=options_features,
        cache_name=cache_name,
        verbose=verbose,
    ).collect()

    if df_options.is_empty():
        return pl.DataFrame(), []

    if verbose:
        elapsed = datetime.now() - start_time
        print(f"Time taken to fetch options: {elapsed}")
        option_rows = len(df_options)
        option_time_count = df_options['timeMs'].n_unique()
        option_expiry_count = df_options['expiry'].n_unique()
        print(
            f"Options dataset stats -> rows: {option_rows:,}, "
            f"unique timeMs: {option_time_count:,}, unique expiries: {option_expiry_count:,}"
        )

    unique_times = []
    if binning is None:
        unique_times = df_options['timeMs'].unique().sort().to_list()
        if verbose:
            print(f"Passing {len(unique_times):,} unique option timestamps to forwards recipe")

    return df_options, unique_times


def _prepare_forward_recipe(
    forwards_recipe: Callable,
    binning: Optional[str],
    unique_times: list[int],
) -> Callable:
    if binning is None:
        return partial(forwards_recipe, unique_times=unique_times)
    return partial(forwards_recipe, binning=binning)


def _fetch_forward_dataset(
    store,
    recipe_for_derivation: Callable,
    dates: list[date],
    cache_name: str,
    verbose: bool,
) -> pl.DataFrame:
    start_time = datetime.now()
    lf_forwards = store.get_derived(
        recipe_for_derivation,
        dates=dates,
        cache_name=cache_name,
        verbose=verbose,
    )
    df_forwards = lf_forwards.collect()
    if df_forwards.is_empty():
        return pl.DataFrame()

    if verbose:
        elapsed = datetime.now() - start_time
        print(f"Time taken to fetch forwards: {elapsed}")
        forward_rows = len(df_forwards)
        forward_time_count = df_forwards['timeMs'].n_unique()
        print(
            f"Forward dataset stats -> rows: {forward_rows:,}, "
            f"unique curve timestamps: {forward_time_count:,}"
        )
    return df_forwards


def _match_options_with_forwards(
    df_options: pl.DataFrame,
    curves_by_time: dict[int, object],
    reconstruct_fn: Callable,
    verbose: bool,
    setup_reference: Optional[datetime] = None,
) -> pl.DataFrame:
    n_rows = len(df_options)
    if n_rows == 0 or not curves_by_time:
        return pl.DataFrame()

    F_bid_array = np.full(n_rows, np.nan, dtype=np.float64)
    F_ask_array = np.full(n_rows, np.nan, dtype=np.float64)

    time_values = df_options['timeMs'].to_numpy()
    T_values = df_options['T'].to_numpy()

    match_start = datetime.now()
    sort_idx = np.argsort(time_values, kind='mergesort')
    sorted_times = time_values[sort_idx]
    change_points = np.flatnonzero(np.diff(sorted_times)) + 1 if len(sorted_times) > 1 else np.array([], dtype=int)
    start_indices = np.concatenate(([0], change_points))
    end_indices = np.concatenate((change_points, [len(sorted_times)]))

    group_count = len(start_indices)

    def _process_group_range(range_start: int, range_end: int) -> tuple[int, int]:
        processed = 0
        missing_curve = 0
        for group_idx in range(range_start, range_end):
            processed += 1
            start_idx = start_indices[group_idx]
            end_idx = end_indices[group_idx]
            row_indices = sort_idx[start_idx:end_idx]
            if len(row_indices) == 0:
                continue

            timeMs = int(sorted_times[start_idx])
            curve = curves_by_time.get(timeMs)
            if curve is None:
                missing_curve += 1
                continue

            option_T = T_values[row_indices]
            if option_T.size == 0:
                continue

            unique_T, inverse_indices = np.unique(option_T, return_inverse=True)
            if unique_T.size == 0:
                continue

            F_bids_unique, F_asks_unique = reconstruct_fn(curve, unique_T)
            F_bids_unique = np.atleast_1d(np.asarray(F_bids_unique, dtype=np.float64))
            F_asks_unique = np.atleast_1d(np.asarray(F_asks_unique, dtype=np.float64))

            F_bid_array[row_indices] = F_bids_unique[inverse_indices]
            F_ask_array[row_indices] = F_asks_unique[inverse_indices]
        return processed, missing_curve

    missing_curve_groups = 0
    if group_count > 0:
        max_workers = min(32, max(1, (os.cpu_count() or 4) - 1))
        chunk_size = max(1, group_count // (max_workers * 8) or 1)
        chunk_size = min(chunk_size, 50_000)
        progress = None
        if verbose:
            progress = tqdm(
                total=group_count,
                desc="Matching forwards to options",
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: list = []
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
    if verbose and missing_curve_groups:
        print(f"Skipped {missing_curve_groups:,} timestamp groups with no forward curve")

    df_options = df_options.with_columns([
        pl.Series('F_bid', F_bid_array),
        pl.Series('F_ask', F_ask_array),
    ])

    df_options = df_options.with_columns(
        (pl.col('strike') / ((pl.col('F_bid') + pl.col('F_ask')) / 2)).alias('moneyness')
    )

    if verbose:
        time_3 = datetime.now()
        matched_mask = np.isfinite(F_bid_array) & np.isfinite(F_ask_array)
        matched_count = int(matched_mask.sum())
        setup_elapsed = ""
        if setup_reference is not None:
            setup_elapsed = f", incl. setup {time_3 - setup_reference}"
        print(
            f"Time taken to match forwards to options: core {time_3 - match_start}{setup_elapsed} "
            f"(matched {matched_count:,} / {n_rows:,} options)"
        )

    return df_options

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

def _build_cache_name(forwards_recipe: Callable, binning: Optional[str] = None) -> str:
    """Build cache name like '{binning|full}_{param1=val1_}forwards' for forwards recipe."""
    recipe_name = _get_recipe_name(forwards_recipe)
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
    recipe_part = recipe_name if not params_str else f"{recipe_name}_{params_str}"
    return f"{prefix}_{recipe_part}_forwards"

# =============================================================================
# Options data preparation
# =============================================================================

def prepare_options(
    store,
    inst_family: str,
    dates: list[date],
    forwards_recipe: Callable,
    binning: Optional[str] = None,
    min_time_to_expiry_hours: float = 2.0,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Prepare options data with metadata and fitted forward prices.
    
    This function:
    - Loads options orderbook data
    - Adds strike, opt_type columns
    - Builds forward curves using the provided recipe (via get_derived for caching)
    - Adds F_bid, F_ask columns for each option
    - Calculates moneyness as strike divided by the bid/ask average (not stored)
    - Returns ALL options (calls and puts, paired or unpaired)
    
    Args:
        store: OrderbookStore instance
        inst_family: Instrument family (e.g., 'BTC-USD')
        dates: List of dates to load
        forwards_recipe: Pre-configured recipe function (use functools.partial)
        binning: Binning interval ('1m', '5m', etc) or None for unbinned
        min_time_to_expiry_hours: Minimum time to expiry (default: 2.0)
        verbose: Whether to print progress (default: True)
    Returns:
        DataFrame with columns:
            - timeMs, symbol, expiry, T
            - bid_1_px, ask_1_px
            - strike, opt_type ('C' or 'P')
            - F_bid, F_ask (from forward curve)
            - moneyness (strike / ((F_bid + F_ask)/2))
    """
    df_options, unique_times = _fetch_options_dataset(
        store=store,
        inst_family=inst_family,
        dates=dates,
        binning=binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        verbose=verbose,
    )
    if df_options.is_empty():
        return pl.DataFrame()

    curve_type, reconstruct_fn = _get_reconstruct_fn(forwards_recipe)
    recipe_for_derivation = _prepare_forward_recipe(
        forwards_recipe=forwards_recipe,
        binning=binning,
        unique_times=unique_times,
    )
    cache_name = _build_cache_name(forwards_recipe, binning)
    df_forwards = _fetch_forward_dataset(
        store=store,
        recipe_for_derivation=recipe_for_derivation,
        dates=dates,
        cache_name=cache_name,
        verbose=verbose,
    )
    if df_forwards.is_empty():
        return pl.DataFrame()

    lookup_start = datetime.now()
    curves_by_time = _build_curve_lookup(df_forwards, curve_type)
    lookup_end = datetime.now()
    if verbose:
        print(
            f"Built forward lookup for {len(curves_by_time):,} timestamps in {lookup_end - lookup_start}"
        )
    if not curves_by_time:
        return pl.DataFrame()

    return _match_options_with_forwards(
        df_options=df_options,
        curves_by_time=curves_by_time,
        reconstruct_fn=reconstruct_fn,
        verbose=verbose,
        setup_reference=lookup_end,
    )


# =============================================================================
# Main options comparison recipe
# =============================================================================

def build_forwards_options_comparison(
    store,
    dates: list[date],
    forwards_recipe: Callable,
    inst_family: str = 'BTC-USD',
    binning: Optional[str] = None,
    min_moneyness: float = 0.9,
    max_moneyness: float = 1.1,
    min_time_to_expiry_hours: float = 2.0,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compare fitted forwards to options-implied forwards using put-call parity.
    
    Automatically detects and handles both PCHIP and Kalman curve types by
    inspecting the DataFrame structure returned by the forwards recipe.
    
    Args:
        store: OrderbookStore instance
        dates: List of dates to evaluate
        forwards_recipe: Pre-configured recipe function (use functools.partial)
                        Examples: partial(build_forwards_pchip, tau_ewma_minutes=5.0)
                                 partial(build_forwards_kalman, lambda_ns=1.0)
        inst_family: Instrument family (default: 'BTC-USD')
        binning: Binning interval ('1m', '5m', etc) or None to use all unique
                option timestamps. If None, will pass unique_times to recipe.
        min_moneyness: Minimum strike/spot ratio to include (default: 0.9)
        max_moneyness: Maximum strike/spot ratio to include (default: 1.1)
        min_time_to_expiry_hours: Minimum time to expiry to include (default: 2.0)
        
    Returns:
        DataFrame with columns:
            - timeMs: Observation timestamp
            - expiry_dt: Option expiry timestamp (ms)
            - strike: Strike price (USD)
            - T: Time to maturity (years)
            - moneyness: strike / ((F_bid + F_ask)/2)
            - call_bid_1_px, call_ask_1_px: Call option prices (USD)
            - put_bid_1_px, put_ask_1_px: Put option prices (USD)
            - F_bid, F_ask, F_mid: Fitted forward prices (USD)
            - F_implied_bid, F_implied_ask, F_implied_mid: Implied forwards from put-call parity (USD)
            - error_bid_bps, error_ask_bps, error_mid_bps: Errors in basis points
            - call_spread_bps, put_spread_bps: Option bid-ask spreads in bps
            
    Examples:
        >>> from functools import partial
        >>> from okx.recipes.forwards import build_forwards_pchip
        >>> from okx.recipes.options_eval import build_forwards_options_comparison
        >>> 
        >>> # Configure forward recipe with partial
        >>> pchip_recipe = partial(build_forwards_pchip, tau_ewma_minutes=5.0)
        >>> 
        >>> # Compare to options-implied forwards
        >>> df_comp = build_forwards_options_comparison(
        ...     store, dates, pchip_recipe,
        ...     binning='5m',
        ...     min_moneyness=0.95, max_moneyness=1.05
        ... )
        >>> 
        >>> # Analyze results
        >>> print(df_comp['error_mid_bps'].describe())
        >>> print(df_comp.group_by('expiry_dt')['error_mid_bps'].mean())
    """
    
    # =========================================================================
    # Step 1: Prepare options with fitted forwards
    # =========================================================================
    
    df_options = prepare_options(
        store=store,
        inst_family=inst_family,
        dates=dates,
        forwards_recipe=forwards_recipe,
        binning=binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        verbose=verbose,
    )
    
    if df_options.is_empty():
        return pl.DataFrame()
    
    # =========================================================================
    # Step 2: Filter by moneyness and create call-put pairs
    # =========================================================================
    
    # Filter by moneyness
    df_options = df_options.filter(
        (pl.col('moneyness') >= min_moneyness) & 
        (pl.col('moneyness') <= max_moneyness)
    )
    
    if df_options.is_empty():
        return pl.DataFrame()
    
    # Separate calls and puts
    df_calls = df_options.filter(pl.col('opt_type') == 'C').select([
        'timeMs', 'expiry', 'strike', 'T', 'moneyness',
        'bid_1_px', 'ask_1_px', 'F_bid', 'F_ask'
    ]).rename({
        'bid_1_px': 'call_bid_1_px',
        'ask_1_px': 'call_ask_1_px',
    })
    
    df_puts = df_options.filter(pl.col('opt_type') == 'P').select([
        'timeMs', 'expiry', 'strike', 'bid_1_px', 'ask_1_px'
    ]).rename({
        'bid_1_px': 'put_bid_1_px',
        'ask_1_px': 'put_ask_1_px',
    })
    
    # Inner join on timeMs, expiry, strike to create matched pairs
    df_pairs = df_calls.join(df_puts, on=['timeMs', 'expiry', 'strike'], how='inner')
    
    if df_pairs.is_empty():
        return pl.DataFrame()
    
    # =========================================================================
    # Step 3: Compute implied forwards and errors
    # =========================================================================
    
    # Extract arrays for vectorized computation
    strikes = df_pairs['strike'].to_numpy()
    
    call_bids = df_pairs['call_bid_1_px'].to_numpy()
    call_asks = df_pairs['call_ask_1_px'].to_numpy()
    put_bids = df_pairs['put_bid_1_px'].to_numpy()
    put_asks = df_pairs['put_ask_1_px'].to_numpy()
    
    F_bids = df_pairs['F_bid'].to_numpy()
    F_asks = df_pairs['F_ask'].to_numpy()
    
    # Skip invalid prices
    valid_mask = ~(
        np.isnan(call_bids) | np.isnan(call_asks) | 
        np.isnan(put_bids) | np.isnan(put_asks) |
        (call_bids <= 0) | (call_asks <= 0) |
        (put_bids <= 0) | (put_asks <= 0) |
        np.isnan(F_bids) | np.isnan(F_asks)
    )
    
    if not np.any(valid_mask):
        return pl.DataFrame()
    
    # Compute implied forwards from put-call parity
    # F ≈ K + (C - P)
    # Conservative bid-ask handling:
    F_implied_bids = strikes + (call_bids - put_asks)
    F_implied_asks = strikes + (call_asks - put_bids)
    F_implied_mids = (F_implied_bids + F_implied_asks) / 2
    
    F_mids = (F_bids + F_asks) / 2
    
    # Compute errors in bps
    error_bids_bps = (np.log(F_implied_bids) - np.log(F_bids)) * 10000
    error_asks_bps = (np.log(F_implied_asks) - np.log(F_asks)) * 10000
    error_mids_bps = (np.log(F_implied_mids) - np.log(F_mids)) * 10000
    
    # Compute option spreads in bps
    call_spread_bps = (call_asks - call_bids) / ((call_asks + call_bids) / 2) * 10000
    put_spread_bps = (put_asks - put_bids) / ((put_asks + put_bids) / 2) * 10000
    
    valid_series = pl.Series('valid_mask', valid_mask)
    
    # Add computed columns to df_pairs
    df_result = (
        df_pairs.with_columns([
            pl.Series('F_mid', F_mids),
            pl.Series('F_implied_bid', F_implied_bids),
            pl.Series('F_implied_ask', F_implied_asks),
            pl.Series('F_implied_mid', F_implied_mids),
            pl.Series('error_bid_bps', error_bids_bps),
            pl.Series('error_ask_bps', error_asks_bps),
            pl.Series('error_mid_bps', error_mids_bps),
            pl.Series('call_spread_bps', call_spread_bps),
            pl.Series('put_spread_bps', put_spread_bps),
            valid_series,
        ])
        .rename({'expiry': 'expiry_dt'})
        .filter(pl.col('valid_mask'))
        .drop('valid_mask')
    )
    
    return df_result
