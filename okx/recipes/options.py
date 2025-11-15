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

CONTRACT_MULTIPLIERS = {
    'BTC-USD': 0.01,
    'ETH-USD': 0.1,
}

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
    1. Loads options orderbook data and adds strike, opt_type columns
    2. Fetches SPOT data and converts option prices from asset units to USD
    3. Uses assign_forwards() to add F_bid, F_ask columns
    4. Calculates moneyness as log(strike / forward_mid)
    
    Returns ALL options (calls and puts, paired or unpaired).
    
    Args:
        store: OrderbookStore instance
        inst_family: Instrument family ('BTC-USD' or 'ETH-USD')
        dates: List of dates to load
        forwards_recipe: Pre-configured recipe function (use functools.partial)
        binning: Binning interval ('1m', '5m', etc) or None for unbinned
        min_time_to_expiry_hours: Minimum time to expiry (default: 2.0)
        verbose: Whether to print progress (default: True)
    
    Returns:
        DataFrame with columns:
            - timeMs, symbol, expiry, T
            - bid_1_px, ask_1_px (in USD)
            - strike, opt_type ('C' or 'P')
            - F_bid, F_ask (from forward curve)
            - moneyness (log(strike / forward_mid))
    """

    # Prepare shared parameters as a dict to be passed to various fetchers or processing steps
    shared_params = {
        'inst_family': inst_family,
        'dates': dates,
        'binning': binning,
        'verbose': verbose,
    }
    # =============================================================================
    # Step 1: Fetch and preprocess options dataset
    # =============================================================================
    start_time = datetime.now()
    cache_name = 'full_options' if binning is None else f'{binning}_options'

    options_features = ['trim', 'strip', 'nullify']
    if binning:
        options_features.extend(['bin', 'sink_bins'])
    else:
        options_features.extend(['dedupe'])
    options_features.extend(['tenor', early_roll(min_time_to_expiry_hours), 'parse_option'])

    df_options = store.get(
        inst_type='OPTION',
        depth=1,
        features=options_features,
        cache_name=cache_name,
        batch_days=3,
        **shared_params,
    ).collect()

    if df_options.is_empty():
        return pl.DataFrame()

    if verbose:
        time_1 = datetime.now()
        print(f"Time taken to fetch options: {time_1 - start_time}")

    # =============================================================================
    # Step 2: Fetch SPOT data and handle numeraire conversion
    # =============================================================================
    contract_multiplier = CONTRACT_MULTIPLIERS[inst_family]
    
    spot_features = ['trim', 'strip', 'nullify']
    if binning:
        spot_features.extend(['bin', 'sink_bins'])
    else:
        spot_features.extend(['dedupe'])
    
    spot_cache_name = 'full_spot' if binning is None else f'{binning}_spot'
    
    df_spot = store.get(
        inst_type='SPOT',
        depth=0,
        features=spot_features,
        cache_name=spot_cache_name,
        batch_days=3,
        **shared_params,
    ).collect()
    
    if df_spot.is_empty():
        raise ValueError(f"No SPOT data found for {inst_family} on {dates}")
    
    if verbose:
        time_2 = datetime.now()
        print(f"Time taken to fetch spot: {time_2 - time_1}")
    
    # Calculate spot mid price and match to option timestamps
    df_spot = df_spot.rename({'mid': 'spot_mid'})
    
    # Join spot prices to options data
    df_options = df_options.join(df_spot, on='timeMs', how='left')
    
    # Convert option prices from asset units to USD
    df_options = df_options.with_columns([
        (pl.col('bid_1_px') * contract_multiplier * pl.col('spot_mid')).alias('bid_1_px'),
        (pl.col('ask_1_px') * contract_multiplier * pl.col('spot_mid')).alias('ask_1_px'),
    ]).drop('spot_mid')
    
    if verbose:
        time_3 = datetime.now()
        print(f"Time taken for numeraire conversion: {time_3 - time_2}")

    # =============================================================================
    # Step 3: Assign forwards using the forward assignment recipe
    # =============================================================================
    df_options = assign_forwards(
        df_data=df_options,
        store=store,
        forwards_recipe=forwards_recipe,
        **shared_params,
    )

    if df_options.is_empty():
        return pl.DataFrame()

    # =============================================================================
    # Step 4: Calculate moneyness
    # =============================================================================
    df_options = df_options.with_columns(
        (pl.col('strike') / ((pl.col('F_bid') + pl.col('F_ask')) / 2)).log().alias('moneyness')
    )

    return df_options