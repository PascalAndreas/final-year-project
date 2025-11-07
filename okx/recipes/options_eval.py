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


# =============================================================================
# Helper functions for options preprocessing
# =============================================================================

def _add_option_metadata(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add strike, opt_type, and moneyness columns to options data.
    
    Expects: symbol, spot_ref columns
    Returns: adds strike, opt_type, moneyness columns
    """
    from okx.helpers import parse_option_name
    
    def extract_strike(symbol: str):
        try:
            _, _, strike, _ = parse_option_name(symbol)
            return strike
        except:
            return None
    
    def extract_opt_type(symbol: str):
        try:
            _, _, _, opt_type = parse_option_name(symbol)
            return opt_type
        except:
            return None
    
    df = lf.collect()
    df = df.with_columns([
        pl.col('symbol').map_elements(extract_strike, return_dtype=pl.Int64).alias('strike'),
        pl.col('symbol').map_elements(extract_opt_type, return_dtype=pl.String).alias('opt_type'),
    ])
    
    # Add moneyness if spot_ref exists
    if 'spot_ref' in df.columns:
        df = df.with_columns(
            (pl.col('strike') / pl.col('spot_ref')).alias('moneyness')
        )
    
    return df.lazy()


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
    cache_name_suffix: str = "_options",
) -> pl.DataFrame:
    """
    Prepare options data with spot reference, metadata, and fitted forward prices.
    
    This function:
    - Loads options orderbook data
    - Joins with SWAP for spot reference
    - Adds strike, opt_type, moneyness columns
    - Builds forward curves using the provided recipe
    - Adds F_fitted_bid, F_fitted_ask columns for each option
    - Returns ALL options (calls and puts, paired or unpaired)
    
    Args:
        store: OrderbookStore instance
        inst_family: Instrument family (e.g., 'BTC-USD')
        dates: List of dates to load
        forwards_recipe: Pre-configured recipe function (use functools.partial)
        binning: Binning interval ('1m', '5m', etc) or None to use unique option timestamps
        min_time_to_expiry_hours: Minimum time to expiry (default: 2.0)
        cache_name_suffix: Suffix for cache name (default: '_options')
        
    Returns:
        DataFrame with columns:
            - timeMs, symbol, expiry, T
            - bid_1_px, ask_1_px
            - spot_ref (from SWAP)
            - strike, opt_type ('C' or 'P'), moneyness
            - F_fitted_bid, F_fitted_ask (from forward curve)
    """
    # Load options orderbook
    cache_name = f'1{cache_name_suffix}'
    
    lf_options = store.get(
        inst_family=inst_family,
        inst_type='OPTION',
        dates=dates,
        depth=1,
        features=['trim', 'tenor'],
        cache_name=cache_name,
    )
    
    df_options = lf_options.collect()
    
    if df_options.is_empty():
        return pl.DataFrame()
    
    # Filter by time to expiry
    min_seconds = min_time_to_expiry_hours * 3600
    df_options = df_options.filter(
        ((pl.col('expiry') - pl.col('timeMs')) / 1000.0) >= min_seconds
    )
    
    if df_options.is_empty():
        return pl.DataFrame()
    
    # Load SWAP for spot reference
    lf_swap = store.get(
        inst_family=inst_family,
        inst_type='SWAP',
        dates=dates,
        depth=1,
        features=['trim', 'mid'],
        cache_name='1_mid',
    )
    
    df_swap = lf_swap.collect()
    
    if df_swap.is_empty():
        return pl.DataFrame()
    
    # Join with spot reference (both must be sorted for join_asof)
    df_swap = df_swap.sort('timeMs').select(['timeMs', 'mid']).rename({'mid': 'spot_ref'})
    df_options = df_options.sort('timeMs').join_asof(df_swap, on='timeMs', strategy='backward')
    
    # Filter out rows without spot reference
    df_options = df_options.filter(pl.col('spot_ref').is_not_null())
    
    # Add strike, opt_type, and moneyness
    df_options = _add_option_metadata(df_options.lazy()).collect()
    
    # Filter out failed parses
    df_options = df_options.filter(
        pl.col('strike').is_not_null() & pl.col('opt_type').is_not_null()
    )
    
    if df_options.is_empty():
        return pl.DataFrame()
    
    # =========================================================================
    # Build forward curves and add to options
    # =========================================================================
    
    # Get unique timestamps from options data
    unique_times = df_options['timeMs'].unique().sort().to_list()
    
    # Build forward curves
    if binning is None:
        lf_forwards = forwards_recipe(
            store,
            dates=dates,
            inst_family=inst_family,
            binning=None,
            unique_times=unique_times,
        )
    else:
        lf_forwards = forwards_recipe(
            store,
            dates=dates,
            inst_family=inst_family,
            binning=binning,
        )
    
    df_forwards = lf_forwards.collect()
    
    if df_forwards.is_empty():
        return pl.DataFrame()
    
    # Add forward prices to each option row
    results = []
    df_forwards_sorted = df_forwards.sort(['timeMs', 'T']) if 'T' in df_forwards.columns else df_forwards
    
    for timeMs in df_options['timeMs'].unique().to_list():
        # Get curve for this timestamp
        df_curve = df_forwards_sorted.filter(pl.col('timeMs') == timeMs)
        
        if df_curve.is_empty():
            continue
        
        # Get all options at this timestamp
        options_at_time = df_options.filter(pl.col('timeMs') == timeMs)
        
        # Extract T values for vectorized forward reconstruction
        T_values = options_at_time['T'].to_numpy()
        
        try:
            # Reconstruct forwards for all T values at once
            F_fitted_bids, F_fitted_asks = _detect_and_reconstruct_forward(df_curve, T_values)
            
            # Add forward prices to options
            options_with_forwards = options_at_time.with_columns([
                pl.Series('F_fitted_bid', F_fitted_bids),
                pl.Series('F_fitted_ask', F_fitted_asks),
            ])
            
            results.append(options_with_forwards)
            
        except Exception:
            # Skip this timestamp on error
            continue
    
    if not results:
        return pl.DataFrame()
    
    return pl.concat(results)


def _detect_and_reconstruct_forward(df_curve: pl.DataFrame, T_target: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Detect curve type from DataFrame columns and reconstruct forward price.
    
    Automatically handles both PCHIP curves and Kalman NS states by inspecting
    the DataFrame column structure.
    
    Args:
        df_curve: Forward curve DataFrame (single timestamp)
        T_target: Target maturity in years (scalar or array)
        
    Returns:
        (F_bid, F_ask) tuple at target maturity
        
    Notes:
        - PCHIP curves have: timeMs, T, F_bid, F_ask, symbol, ln_F_bid, ln_F_ask
        - Kalman states have: timeMs, beta0, beta1, beta2, lambda_ns, ln_F_ref_bid, ln_F_ref_ask
    """
    if 'F_bid' in df_curve.columns and 'T' in df_curve.columns:
        # PCHIP curve - has pillar points, need to interpolate
        from forwards.pchip import PCHIPCurve, reconstruct_forward
        
        curve = PCHIPCurve(
            timeMs=int(df_curve['timeMs'][0]),
            T_nodes=df_curve['T'].to_numpy(),
            ln_F_bid_nodes=np.log(df_curve['F_bid'].to_numpy()),
            ln_F_ask_nodes=np.log(df_curve['F_ask'].to_numpy()),
            symbols=df_curve['symbol'].to_list(),
        )
        return reconstruct_forward(curve, T_target)
        
    elif 'beta0' in df_curve.columns:
        # Kalman state - has NS parameters, need to reconstruct
        from forwards.kalman_ns import NSCarryState, reconstruct_ns_forward
        
        state = NSCarryState.from_polars(df_curve)
        F_bid = reconstruct_ns_forward(state, T_target, use_bid=True)
        F_ask = reconstruct_ns_forward(state, T_target, use_bid=False)
        return F_bid, F_ask
        
    else:
        raise ValueError(
            f"Unknown curve format. Expected PCHIP (F_bid, T) or Kalman (beta0, beta1, beta2). "
            f"Got columns: {df_curve.columns}"
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
            - spot_ref: Reference spot price from SWAP (USD)
            - moneyness: strike / spot_ref
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
        'timeMs', 'expiry', 'strike', 'T', 'spot_ref', 'moneyness',
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
    
    # Add computed columns to df_pairs
    df_result = df_pairs.with_columns([
        pl.Series('F_fitted_mid', F_fitted_mids),
        pl.Series('F_implied_bid', F_implied_bids),
        pl.Series('F_implied_ask', F_implied_asks),
        pl.Series('F_implied_mid', F_implied_mids),
        pl.Series('error_bid_bps', error_bids_bps),
        pl.Series('error_ask_bps', error_asks_bps),
        pl.Series('error_mid_bps', error_mids_bps),
        pl.Series('call_spread_bps', call_spread_bps),
        pl.Series('put_spread_bps', put_spread_bps),
    ]).rename({'expiry': 'expiry_dt'})
    
    # Filter to valid rows
    df_result = df_result.filter(pl.Series('valid', valid_mask))
    
    return df_result
