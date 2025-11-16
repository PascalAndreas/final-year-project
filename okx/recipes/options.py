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
from datetime import date, datetime
from typing import Callable, Optional

from okx.recipes.helpers import early_roll
from okx.recipes.forwards import assign_forwards
from okx.helpers import _get_function_name

# =============================================================================
# Options data preparation
# =============================================================================

CONTRACT_MULTIPLIERS = {
    'BTC-USD': 0.01,
    'ETH-USD': 0.1,
}

def _pair_options(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Pair calls and puts with matching timeMs, expiry, and strike."""
    lf_calls = lf.filter(pl.col('opt_type') == 'C').select([
        'timeMs', 'expiry', 'strike', 'T',
        'bid_1_px', 'ask_1_px'
    ]).rename({
        'bid_1_px': 'call_bid_1_px',
        'ask_1_px': 'call_ask_1_px',
    })
    lf_puts = lf.filter(pl.col('opt_type') == 'P').select([
        'timeMs', 'expiry', 'strike', 'bid_1_px', 'ask_1_px'
    ]).rename({
        'bid_1_px': 'put_bid_1_px',
        'ask_1_px': 'put_ask_1_px',
    })
    return lf_calls.join(lf_puts, on=['timeMs', 'expiry', 'strike'], how='inner')

def prepare_options(
    store,
    inst_family: str,
    dates: list[date],
    forwards_recipe: Callable,
    binning: Optional[str] = None,
    batch_days: Optional[int] = None,
    paired: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame:
    """
    Prepare options data with metadata and fitted forward prices.
    
    This function:
    1. Loads options orderbook data and adds strike, opt_type columns
    2. Optionally pairs calls and puts with matching strike/expiry
    3. Fetches SPOT data and converts option prices from asset units to USD
    4. Uses assign_forwards() to add F_bid, F_ask columns
    5. Calculates moneyness as log(strike / forward_mid)
    
    Args:
        store: OrderbookStore instance
        inst_family: Instrument family ('BTC-USD' or 'ETH-USD')
        dates: List of dates to load
        forwards_recipe: Pre-configured recipe function (use functools.partial)
        binning: Binning interval ('1m', '5m', etc) or None for unbinned
        batch_days: Number of days to process in each batch (default: None)
        paired: If True, pair calls and puts with matching strike/expiry (default: False)
        verbose: Whether to print progress (default: True)
    
    Returns:
        LazyFrame with columns:
        
        If paired=False:
            - timeMs, symbol, expiry, T
            - bid_1_px, ask_1_px (in USD)
            - strike, opt_type ('C' or 'P')
            - F_bid, F_ask (from forward curve)
            - moneyness (log(strike / forward_mid))
        
        If paired=True:
            - timeMs, expiry, strike, T
            - call_bid_1_px, call_ask_1_px (in USD)
            - put_bid_1_px, put_ask_1_px (in USD)
            - F_bid, F_ask (from forward curve)
            - moneyness (log(strike / forward_mid))
    """
    if verbose:
        start_time = datetime.now()
        print(f"Preparing {'paired ' if paired else ''}options data for {inst_family} for {len(dates)} dates using {_get_function_name(forwards_recipe).lower()}{f' and {binning} binning' if binning else ''}")
    # Prepare shared parameters as a dict to be passed to various fetchers or processing steps
    shared_params = {
        'inst_family': inst_family,
        'dates': dates,
        'binning': binning,
        'verbose': verbose
    }

    # =============================================================================
    # Step 1: Fetch and preprocess options dataset
    # =============================================================================
    
    cache_name = 'full_options' if binning is None else f'{binning}_options'
    options_features = ['trim', 'strip', 'nullify']
    if binning:
        options_features.extend(['bin', 'sink_bins'])
    else:
        options_features.extend(['dedupe'])
    options_features.extend(['tenor', 'parse_option'])

    lf_options = store.get(
        inst_type='OPTION',
        depth=1,
        features=options_features,
        cache_name=cache_name,
        batch_days=batch_days,
        **shared_params
    )

    # Optionally pair calls and puts early (before numeraire conversion)
    if paired:
        lf_options = _pair_options(lf_options)

    if verbose:
        time_1 = datetime.now()
        print(f" - Time taken to fetch options: {time_1 - start_time}")

    # =============================================================================
    # Step 2: Fetch and preprocess spot data
    # =============================================================================
    
    spot_cache_name = 'full_spot' if binning is None else f'{binning}_spot'
    spot_features = ['trim', 'strip']
    if binning:
        spot_features.extend(['bin', 'sink_bins'])
    else:
        spot_features.extend(['dedupe'])
    spot_features.extend([lambda lf: (lf.rename({'mid': 'spot_mid'}))])
    
    lf_spot = store.get(
        inst_type='SPOT',
        depth=0,
        features=spot_features,
        cache_name=spot_cache_name,
        batch_days=batch_days,
        **shared_params
    )
    
    if verbose:
        time_2 = datetime.now()
        print(f" - Time taken to fetch spot: {time_2 - time_1}")

    # =============================================================================
    # Step 3: Numeraire conversion
    # =============================================================================
    
    if binning:
        # When binning, timestamps are aligned so we can use a regular join
        lf_options = lf_options.join(lf_spot, on='timeMs', how='left')
    else:
        # When not binning, we need asof join to match each option with most recent spot
        lf_options = lf_options.sort('timeMs')
        lf_spot = lf_spot.sort('timeMs')
        lf_options = lf_options.join_asof(lf_spot, on='timeMs', strategy='backward')

    # Convert option prices from asset units to USD using spot mid price
    # Apply to all columns ending in '_px' (works for both paired and unpaired data)
    contract_multiplier = CONTRACT_MULTIPLIERS[inst_family]
    lf_options = lf_options.with_columns(
        (pl.col('^.*_px$') * contract_multiplier * pl.col('spot_mid'))
    ).drop('spot_mid')
    
    if verbose:
        time_3 = datetime.now()
        print(f" - Time taken for numeraire conversion: {time_3 - time_2}")

    # =============================================================================
    # Step 4: Assign forwards and moneyness
    # =============================================================================

    lf_options = assign_forwards(
        lf_data=lf_options,
        store=store,
        forwards_recipe=forwards_recipe,
        **shared_params,
    )
    lf_options = lf_options.with_columns(
        (pl.col('strike') / ((pl.col('F_bid') + pl.col('F_ask')) / 2)).log().alias('moneyness')
    )
    
    if verbose:
        time_4 = datetime.now()
        print(f" - Time taken for forwards assignment and moneyness calculation: {time_4 - time_3}")
        print(f" - Total time taken for options data preparation: {time_4 - start_time}")

    return lf_options