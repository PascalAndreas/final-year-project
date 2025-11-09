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

import polars as pl
import numpy as np
from datetime import date
from typing import Callable, Optional
from functools import partial

from okx.recipes.helpers import early_roll, finalize_binning

from datetime import datetime
from tqdm import tqdm

# =============================================================================
# Helper functions
# =============================================================================

def _get_recipe_name(recipe: Callable) -> str:
    """Extract function name from recipe (handles functools.partial)."""
    if isinstance(recipe, partial):
        return recipe.func.__name__
    return recipe.__name__

def _reconstruct_forward_pchip(df_curve: pl.DataFrame, T_target: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Reconstruct forward from PCHIP curve."""
    from forwards.pchip import PCHIPCurve, reconstruct_forward
    
    curve = PCHIPCurve.from_polars(df_curve)
    return reconstruct_forward(curve, T_target)

def _reconstruct_forward_kalman(df_curve: pl.DataFrame, T_target: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Reconstruct forward from Kalman NS state."""
    from forwards.kalman_ns import NSCarryState, reconstruct_ns_forward
    
    state = NSCarryState.from_polars(df_curve)
    F_bid = reconstruct_ns_forward(state, T_target, use_bid=True)
    F_ask = reconstruct_ns_forward(state, T_target, use_bid=False)
    return F_bid, F_ask

def _get_reconstruct_fn(forwards_recipe: Callable) -> Callable:
    """Get the appropriate reconstruction function based on recipe name."""
    recipe_name = _get_recipe_name(forwards_recipe)
    if 'pchip' in recipe_name.lower():
        return _reconstruct_forward_pchip
    elif 'kalman' in recipe_name.lower():
        return _reconstruct_forward_kalman
    else:
        raise ValueError(f"Unknown recipe type: {recipe_name}")

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
    if isinstance(forwards_recipe, partial):
        recipe_name = _get_recipe_name(forwards_recipe)
        params = forwards_recipe.keywords or {}
        param_parts = [
            f"{key}={_format_cache_value(params[key])}"
            for key in sorted(params)
        ]
        params_str = "_".join(param_parts) + ("_" if param_parts else "")
    else:
        recipe_name = _get_recipe_name(forwards_recipe)
        params_str = ""
    prefix = binning if binning is not None else "full"
    return f"{prefix}_{params_str}forwards"

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
    - Adds F_fitted_bid, F_fitted_ask columns for each option
    - Calculates moneyness as strike / F_fitted_mid
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
            - F_fitted_bid, F_fitted_ask (from forward curve)
            - moneyness (strike / F_fitted_mid)
    """
    start_time = datetime.now()
    # =============================================================================
    # Fetch options orderbook
    # =============================================================================
    
    cache_name = 'full_options' if binning is None else f'{binning}_options'
    
    # Build feature list
    options_features = ['trim', 'strip']
    if binning:
        options_features.extend(['bin', finalize_binning])
    else:
        options_features.extend(['dedupe'])
    options_features.extend(['tenor', early_roll(min_time_to_expiry_hours), 'parse_option'])
    
    # Load options orderbook
    df_options = store.get(
        inst_type='OPTION',
        inst_family=inst_family,
        dates=dates,
        depth=1,
        binning=binning,
        features=options_features,
        cache_name=cache_name,
    ).collect()

    if df_options.is_empty():
        return pl.DataFrame()
    
    time_1 = datetime.now()
    print(f"Time taken to fetch options: {time_1 - start_time}")
    # =========================================================================
    # Build forward curves and add to options
    # =========================================================================
    
    cache_name = _build_cache_name(forwards_recipe, binning)
    
    # Get reconstruction function based on recipe type
    reconstruct_fn = _get_reconstruct_fn(forwards_recipe)
    
    # Handle unique_times for unbinned data, otherwise pass binning to recipe
    if binning is None:
        unique_times = df_options['timeMs'].unique().sort().to_list()
        recipe_for_derivation = partial(forwards_recipe, unique_times=unique_times)
    else:
        recipe_for_derivation = partial(forwards_recipe, binning=binning)
    
    lf_forwards = store.get_derived(recipe_for_derivation, dates=dates, cache_name=cache_name)
    df_forwards = lf_forwards.collect()
    
    if df_forwards.is_empty():
        return pl.DataFrame()
    
    time_2 = datetime.now()
    print(f"Time taken to fetch forwards: {time_2 - time_1}")

    # Add forward prices to each option row
    results = []

    timeMs_list = df_options['timeMs'].unique().sort().to_list()
    timeMs_iter = tqdm(timeMs_list, desc="Matching forwards to options", disable=not verbose)

    for timeMs in timeMs_iter:
        # Get curve for this timestamp
        df_curve = df_forwards.filter(pl.col('timeMs') == timeMs)
        if df_curve.is_empty():
            continue
        
        # Get all options at this timestamp
        options_at_time = df_options.filter(pl.col('timeMs') == timeMs)
        
        # Optimize: Many options share same T (expiry) but differ in strike/type
        # Only reconstruct forwards for unique T values, then map back
        T_values = options_at_time['T'].to_numpy()
        unique_T, inverse_indices = np.unique(T_values, return_inverse=True)
        
        try:
            # Reconstruct forwards only for unique T values
            F_fitted_bids_unique, F_fitted_asks_unique = reconstruct_fn(df_curve, unique_T)
            F_fitted_mids_unique = (F_fitted_bids_unique + F_fitted_asks_unique) / 2
            
            # Map unique results back to all options using inverse indices
            F_fitted_bids = F_fitted_bids_unique[inverse_indices]
            F_fitted_asks = F_fitted_asks_unique[inverse_indices]
            F_fitted_mids = F_fitted_mids_unique[inverse_indices]
            
            # Add forward prices and moneyness to options
            options_with_forwards = options_at_time.with_columns([
                pl.Series('F_fitted_bid', F_fitted_bids),
                pl.Series('F_fitted_ask', F_fitted_asks),
                pl.Series('F_fitted_mid', F_fitted_mids),
            ]).with_columns(
                (pl.col('strike') / pl.col('F_fitted_mid')).alias('moneyness')
            )
            
            results.append(options_with_forwards)
            
        except Exception:
            # Skip this timestamp on error
            continue
    
    if not results:
        return pl.DataFrame()
    
    time_3 = datetime.now()
    print(f"Time taken to add forwards to options: {time_3 - time_2}")
    
    return pl.concat(results)


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
            - moneyness: strike / F_fitted_mid
            - call_bid_1_px, call_ask_1_px: Call option prices (USD)
            - put_bid_1_px, put_ask_1_px: Put option prices (USD)
            - F_fitted_bid, F_fitted_ask, F_fitted_mid: Fitted forward prices (USD)
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
        'bid_1_px', 'ask_1_px', 'F_fitted_bid', 'F_fitted_ask'
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
    
    F_fitted_bids = df_pairs['F_fitted_bid'].to_numpy()
    F_fitted_asks = df_pairs['F_fitted_ask'].to_numpy()
    
    # Skip invalid prices
    valid_mask = ~(
        np.isnan(call_bids) | np.isnan(call_asks) | 
        np.isnan(put_bids) | np.isnan(put_asks) |
        (call_bids <= 0) | (call_asks <= 0) |
        (put_bids <= 0) | (put_asks <= 0) |
        np.isnan(F_fitted_bids) | np.isnan(F_fitted_asks)
    )
    
    if not np.any(valid_mask):
        return pl.DataFrame()
    
    # Compute implied forwards from put-call parity
    # F ≈ K + (C - P)
    # Conservative bid-ask handling:
    F_implied_bids = strikes + (call_bids - put_asks)
    F_implied_asks = strikes + (call_asks - put_bids)
    F_implied_mids = (F_implied_bids + F_implied_asks) / 2
    
    F_fitted_mids = (F_fitted_bids + F_fitted_asks) / 2
    
    # Compute errors in bps
    error_bids_bps = (np.log(F_implied_bids) - np.log(F_fitted_bids)) * 10000
    error_asks_bps = (np.log(F_implied_asks) - np.log(F_fitted_asks)) * 10000
    error_mids_bps = (np.log(F_implied_mids) - np.log(F_fitted_mids)) * 10000
    
    # Compute option spreads in bps
    call_spread_bps = (call_asks - call_bids) / ((call_asks + call_bids) / 2) * 10000
    put_spread_bps = (put_asks - put_bids) / ((put_asks + put_bids) / 2) * 10000
    
    valid_series = pl.Series('valid_mask', valid_mask)
    
    # Add computed columns to df_pairs
    df_result = (
        df_pairs.with_columns([
            pl.Series('F_fitted_mid', F_fitted_mids),
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
