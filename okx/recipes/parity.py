import polars as pl
from datetime import date, datetime
from typing import Optional
from okx.recipes.helpers import pair_options
from okx.fees import get_fees, get_options_fees
from okx.features import sink_bins

# =============================================================================
# Market Snapshots
# =============================================================================

def prepare_market_snapshots(
    store,
    inst_family: str,
    dates: list[date],
    binning: str,
    verbose: bool = False,
    batch_days: int = 5,
) -> pl.LazyFrame:
    """
    Prepare market snapshots for parity evaluation.
    """
    if verbose:
        start_time = datetime.now()
        print(f"Constructing market snapshots for {inst_family} with {binning} binning")
    # Prepare shared parameters
    shared_params = {
        'inst_family': inst_family,
        'dates': dates,
        'binning': binning,
        'verbose': verbose,
        'batch_days': batch_days,
        'cache_name': f'{binning}_market_snapshots'
    }

    # Fetch futures data
    futures_features = ['trim', 'strip', 'bin_ff', 'sink_bins', 'mid', 'tenor']
    futures_lf = store.get(
        inst_type='FUTURES',
        depth=1,
        **shared_params,
        features=futures_features,
    ).rename({'bid_1_px': 'F_bid', 'ask_1_px': 'F_ask', 'mid': 'F_mid'}).drop('symbol')

    # Fetch options data
    options_features = ['trim', 'bin_ff', 'sink_bins', 'nullify', 'drop_nulls_strict', 'parse_option', 'tenor']
    options_lf = store.get(
        inst_type='OPTION',
        depth=1,
        **shared_params,
        features=options_features,
    )

    # Pair options
    options_lf = pair_options(options_lf, include_qty=True)

    # Join and add moneyness
    merged = futures_lf.join(options_lf, on=['timeMs', 'expiry', 'T'], how='inner')
    merged = merged.with_columns((pl.col('strike') / pl.col('F_mid')).log().alias('moneyness'))

    # Fetch spot data
    spot_features = ['trim', 'strip', 'bin_ff', 'sink_bins']
    spot_lf = store.get(
        inst_type='SPOT',
        depth=0,
        **shared_params,
        features=spot_features,
    ).drop('symbol').rename({'mid': 'BTC-USD'})

    # Join spot to add BTC-USD column
    merged = merged.join(spot_lf, on=['timeMs'], how='left')

    if verbose:
        end_time = datetime.now()
        print(f" - Time taken to construct market snapshots: {end_time - start_time}")

    return merged

# =============================================================================
# Parity Arbitrage using listed futures contracts
# =============================================================================

def calculate_futures_parity_arbitrage(
    store,
    inst_family: str,
    dates: list[date],
    binning: str,
    assets: float = 0,
    volume: float = 0,
    max_return: float = 10.0,
    verbose: bool = False,
    batch_days: int = 5,
) -> pl.LazyFrame:
    """
    Calculate put-call parity arbitrage opportunities from market snapshots.
    
    NOTE: Assets and Volume bounds for options fee table are in USD, while the futures fee table bounds are in EUR.
          For now, we're ignoring this discrepancy.

    Args:
        store: Data store object
        inst_family: Instrument family (e.g., 'BTC-USD')
        dates: List of dates to process
        binning: Binning interval (e.g., '10s', '10m')
        assets: Account assets in USD for options and futures fee tiers
        volume: 30-day trading volume in USD for options and futures fee tiers
        max_return: Maximum return to clip outliers (default: 10.0 = 1000%)
        verbose: Whether to print progress
        batch_days: Number of days to process in each batch
        
    Returns:
        LazyFrame with synthetic forwards, arbitrage metrics for each order type

    TODO: Capital costs don't include the cost of the futures contract for trades which are long listed futures.
    """
    if verbose:
        start_time = datetime.now()
        print("Calculating parity arbitrage opportunities...")
    
    # Prepare market snapshots
    lf = prepare_market_snapshots(
        store=store,
        inst_family=inst_family,
        dates=dates,
        binning=binning,
        verbose=verbose,
        batch_days=batch_days,
    )
    
    # Get fee rates
    options_tier, options_maker_fee, options_taker_fee = get_options_fees(assets, volume)
    futures_tier, futures_maker_fee, futures_taker_fee = get_fees(assets, volume)
    
    if verbose:
        print(f"  {options_tier:>8} options fees: maker={options_maker_fee:.4%}, taker={options_taker_fee:.4%}")
        print(f"  {futures_tier:>8} futures fees: maker={futures_maker_fee:.4%}, taker={futures_taker_fee:.4%}")
    
    # Define synthetic order types: (name, direction, call_order, put_order, description)
    order_defs = [
        # LONG synthetics (buy call, sell put)
        ('LONG_LMT_Cb_Pa', 'long', 'maker', 'maker', 'Long limit: buy call @ bid, sell put @ ask'),
        ('LONG_SYM_Cb_Pb', 'long', 'maker', 'taker', 'Long sym: buy call @ bid, sell put @ bid'),
        ('LONG_SYM_Ca_Pa', 'long', 'taker', 'maker', 'Long sym: buy call @ ask, sell put @ ask'),
        ('LONG_MKT_Ca_Pb', 'long', 'taker', 'taker', 'Long market: buy call @ ask, sell put @ bid'),
        # SHORT synthetics (sell call, buy put)
        ('SHORT_LMT_Ca_Pb', 'short', 'maker', 'maker', 'Short limit: sell call @ ask, buy put @ bid'),
        ('SHORT_SYM_Ca_Pa', 'short', 'maker', 'taker', 'Short sym: sell call @ ask, buy put @ ask'),
        ('SHORT_SYM_Cb_Pb', 'short', 'taker', 'maker', 'Short sym: sell call @ bid, buy put @ bid'),
        ('SHORT_MKT_Cb_Pa', 'short', 'taker', 'taker', 'Short market: sell call @ bid, buy put @ ask'),
    ]
    
    # Calculate synthetic forwards and arbitrage for each order type
    columns = []
    
    option_tick = 0.0001
    contract_multiplier = 0.01 if inst_family == 'BTC-USD' else 0.1
    inverse_multiplier = 100 if inst_family == 'BTC-USD' else 10
    # If order is maker, we must add/subtract one tick to be at the top of the order book.
    # If order is taker, we are limited in size by the qty of the top bid/ask.
    # If the order is maker, we take qty to be inverse_multiplier contracts.
    for name, direction, call_order, put_order, description in order_defs:
        # Select prices based on order type
        if direction == 'long':
            call_px = (pl.col('call_bid') + option_tick) if call_order == 'maker' else pl.col('call_ask') # Buy call
            put_px = (pl.col('put_ask') - option_tick) if put_order == 'maker' else pl.col('put_bid') # Sell put
            call_qty = inverse_multiplier if call_order == 'maker' else pl.col('call_ask_qty')
            put_qty = inverse_multiplier if put_order == 'maker' else pl.col('put_bid_qty')
            call_fee_rate = options_maker_fee if call_order == 'maker' else options_taker_fee
            put_fee_rate = options_maker_fee if put_order == 'maker' else options_taker_fee
        else:  # short
            call_px = (pl.col('call_ask') - option_tick) if call_order == 'maker' else pl.col('call_bid') # Sell call
            put_px = (pl.col('put_bid') + option_tick) if put_order == 'maker' else pl.col('put_ask') # Buy put
            call_qty = inverse_multiplier if call_order == 'maker' else pl.col('call_bid_qty')
            put_qty = inverse_multiplier if put_order == 'maker' else pl.col('put_ask_qty')
            call_fee_rate = options_maker_fee if call_order == 'maker' else options_taker_fee
            put_fee_rate = options_maker_fee if put_order == 'maker' else options_taker_fee
        
        # Convert prices to USD
        call_amt = call_px * pl.col('BTC-USD')
        put_amt = put_px * pl.col('BTC-USD')
        
        # Position size (limited by option liquidity, in units of contracts)
        position_size = pl.min_horizontal(call_qty, put_qty)
        
        # As confirmed by sanity check, the quotes prices are for one BTC worth of contracts.
        # This means that the premium is for 100 contracts (for BTC-USD) or 10 contracts (for ETH-USD).

        # Calculate net premium
        if direction == 'long':
            net_premium = (call_amt - put_amt)
        else:
            net_premium = (put_amt - call_amt)

        # Calculate fees
        options_fees = (call_amt * call_fee_rate + put_amt * put_fee_rate)
        if direction == 'long':
            futures_fee = pl.col('F_bid') * futures_taker_fee
        else:
            futures_fee = pl.col('F_ask') * futures_taker_fee
        net_capital = net_premium + options_fees + futures_fee
        
        # Calculated quoted synthetic forwards prices and profit (in USD, including both premium and fees).
        if direction == 'long':
            # F_synthetic = K + (C - P)
            synth_fwd = pl.col('strike') + net_capital
            # Long synthetic + short futures
            profit = pl.col('F_bid') - synth_fwd
        else:
            # F_synthetic = K - (P - C)
            synth_fwd = pl.col('strike') - net_capital
            # Short synthetic + long futures
            profit = synth_fwd - pl.col('F_ask')
        
        # Total profit and capital, accounting for contract multiplier and maximimum position size.
        profit = profit * contract_multiplier * position_size
        capital = net_capital * contract_multiplier * position_size
        
        # Returns (NaN when capital <= 0 or division issues)
        raw_return = pl.when(capital > 0).then(profit / capital).otherwise(pl.lit(float('nan')))
        
        # Clip returns to prevent extreme outliers
        raw_return_clipped = raw_return.clip(-max_return, max_return)
        
        # Annualized return
        annual_return = pl.when(raw_return_clipped.is_not_nan()).then((1 + raw_return_clipped).pow(1 / pl.col('T')) - 1).otherwise(pl.lit(float('nan')))
        
        # Profitable flag
        profitable = profit > 0
        
        # Add columns
        columns.extend([
            synth_fwd.alias(f'{name}_fwd'),
            position_size.alias(f'{name}_size'),
            profit.alias(f'{name}_profit'),
            capital.alias(f'{name}_capital'),
            raw_return_clipped.alias(f'{name}_return'),
            annual_return.alias(f'{name}_annual_return'),
            profitable.alias(f'{name}_profitable'),
        ])
    
    lf = lf.with_columns(columns)
    
    if verbose:
        end_time = datetime.now()
        print(f"  Time taken: {end_time - start_time}")
    
    return lf


def summarize_futures_parity_arbitrage(
    lf: pl.LazyFrame,
    order_types: list[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Summarize arbitrage statistics for specified order types.
    
    Args:
        lf: LazyFrame with parity arbitrage metrics
        order_types: List of order type names to summarize (default: all LONG/SHORT types)
        verbose: Whether to print summary
        
    Returns:
        Dictionary with statistics for each order type
    """
    if order_types is None:
        order_types = [
            'LONG_LMT_Cb_Pa', 'LONG_SYM_Cb_Pb', 'LONG_SYM_Ca_Pa', 'LONG_MKT_Ca_Pb',
            'SHORT_LMT_Ca_Pb', 'SHORT_SYM_Ca_Pa', 'SHORT_SYM_Cb_Pb', 'SHORT_MKT_Cb_Pa',
        ]
    
    results = {}
    
    for name in order_types:
        profitable_col = f'{name}_profitable'
        profit_col = f'{name}_profit'
        capital_col = f'{name}_capital'
        return_col = f'{name}_return'
        annual_return_col = f'{name}_annual_return'
        
        stats = lf.select([
            # Count and frequency
            pl.col(profitable_col).sum().alias('count'),
            pl.col(profitable_col).mean().alias('frequency'),
            # Profit stats (when profitable)
            pl.when(pl.col(profitable_col))
              .then(pl.col(profit_col))
              .mean()
              .alias('avg_profit'),
            pl.when(pl.col(profitable_col))
              .then(pl.col(profit_col))
              .max()
              .alias('max_profit'),
            # Capital stats (when profitable)
            pl.when(pl.col(profitable_col))
              .then(pl.col(capital_col))
              .mean()
              .alias('avg_capital'),
            # Return stats (when profitable and not NaN)
            pl.when(pl.col(profitable_col) & pl.col(return_col).is_not_nan())
              .then(pl.col(return_col))
              .mean()
              .alias('avg_return'),
            pl.when(pl.col(profitable_col) & pl.col(annual_return_col).is_not_nan())
              .then(pl.col(annual_return_col))
              .mean()
              .alias('avg_annual_return'),
        ]).collect()
        
        results[name] = {
            'count': stats['count'][0],
            'frequency': stats['frequency'][0],
            'avg_profit': stats['avg_profit'][0],
            'max_profit': stats['max_profit'][0],
            'avg_capital': stats['avg_capital'][0],
            'avg_return': stats['avg_return'][0],
            'avg_annual_return': stats['avg_annual_return'][0],
        }
    
    if verbose:
        print("\n=== PARITY ARBITRAGE SUMMARY ===\n")
        for name, stats in results.items():
            direction = 'LONG' if name.startswith('LONG') else 'SHORT'
            execution = name.split('_')[1]
            print(f"{name} ({direction} {execution}):")
            print(f"  Count: {stats['count']:,} | Frequency: {stats['frequency']:.2%}")
            print(f"  Avg Profit: ${stats['avg_profit']:.2f} | Max: ${stats['max_profit']:.2f}")
            print(f"  Avg Capital: ${stats['avg_capital']:.2f}")
            print(f"  Avg Return: {stats['avg_return']:.2%} | Annual: {stats['avg_annual_return']:.2%}")
            print()
    
    return results

def aggregate_best_futures_synthetics(
    lf: pl.LazyFrame,
    verbose: bool = False,
) -> pl.LazyFrame:
    """
    For each timestamp, find the best synthetic prices across all strikes,
    and include F_mid as a reference mid price for the future.

    Returns simple best prices (no quantity weighting - use the single best opportunity).

    Args:
        lf: LazyFrame with parity arbitrage metrics
        verbose: Whether to print progress

    Returns:
        LazyFrame with best synthetic prices per timestamp, and F_mid
    """
    if verbose:
        print("Aggregating best synthetic prices across strikes...")

    result = lf.group_by('timeMs').agg([
        # Include the mid price of the future
        pl.col('F_mid').first().alias('F_mid'),

        # Market orders: best single price
        pl.col('LONG_MKT_Ca_Pb_fwd').min().alias('best_market_long'),
        pl.col('SHORT_MKT_Cb_Pa_fwd').max().alias('best_market_short'),

        # Symmetric orders: best single price across both types
        pl.min_horizontal(
            pl.col('LONG_SYM_Cb_Pb_fwd').min(),
            pl.col('LONG_SYM_Ca_Pa_fwd').min(),
        ).alias('best_sym_long'),

        pl.max_horizontal(
            pl.col('SHORT_SYM_Ca_Pa_fwd').max(),
            pl.col('SHORT_SYM_Cb_Pb_fwd').max(),
        ).alias('best_sym_short'),
    ]).sort('timeMs')

    if verbose:
        print("  Aggregated best prices per timestamp (including F_mid)")

    return result

def aggregate_best_futures_synthetics_with_levels(
    lf: pl.LazyFrame,
    levels: list[float] = [10, 50, 100],
    verbose: bool = False,
) -> pl.LazyFrame:
    """
    For each timestamp, aggregate the best synthetic prices across strikes.
    Uses vectorized Polars operations for speed.
    
    Args:
        lf: LazyFrame with parity arbitrage metrics
        levels: Contract size levels to aggregate (e.g., [10, 50, 100])
        verbose: Whether to print progress
        
    Returns:
        LazyFrame with best synthetic prices per timestamp and contract level
    """
    if verbose:
        print("Aggregating best synthetic prices across strikes...")
    
    # Helper function to calculate weighted average up to level
    def weighted_avg_expr(fwd_col: str, size_col: str, level: float, ascending: bool = True):
        """Create expression for weighted average up to level contracts."""
        return (
            pl.col(fwd_col).sort_by(pl.col(fwd_col), descending=not ascending)
            .over('timeMs')
            .alias('_sorted_fwd'),
            pl.col(size_col).sort_by(pl.col(fwd_col), descending=not ascending)
            .over('timeMs')
            .alias('_sorted_size'),
        )
    
    # Start with basic aggregations
    agg_exprs = [
        # Best market prices
        pl.col('LONG_MKT_Ca_Pb_fwd').min().alias('best_market_long'),
        pl.col('SHORT_MKT_Cb_Pa_fwd').max().alias('best_market_short'),
    ]
    
    # For symmetric orders, we need to combine opportunities and do weighted averages
    # This is complex, so let's do it in steps
    
    # Create long opportunities dataframe
    long_opps = lf.select([
        'timeMs',
        pl.concat_list([
            pl.struct([
                pl.col('LONG_SYM_Cb_Pb_fwd').alias('fwd'),
                pl.col('LONG_SYM_Cb_Pb_size').alias('size'),
            ]),
            pl.struct([
                pl.col('LONG_SYM_Ca_Pa_fwd').alias('fwd'),
                pl.col('LONG_SYM_Ca_Pa_size').alias('size'),
            ]),
        ]).alias('opportunities')
    ]).explode('opportunities').unnest('opportunities')
    
    # Create short opportunities dataframe
    short_opps = lf.select([
        'timeMs',
        pl.concat_list([
            pl.struct([
                pl.col('SHORT_SYM_Ca_Pa_fwd').alias('fwd'),
                pl.col('SHORT_SYM_Ca_Pa_size').alias('size'),
            ]),
            pl.struct([
                pl.col('SHORT_SYM_Cb_Pb_fwd').alias('fwd'),
                pl.col('SHORT_SYM_Cb_Pb_size').alias('size'),
            ]),
        ]).alias('opportunities')
    ]).explode('opportunities').unnest('opportunities')
    
    # For each level, calculate weighted averages
    level_results = []
    
    for level in levels:
        # LONG: sort by fwd (ascending), take cumulative sum up to level
        long_agg = (
            long_opps
            .sort(['timeMs', 'fwd'])
            .with_columns([
                pl.col('size').cum_sum().over('timeMs').alias('cum_size'),
            ])
            .filter(
                (pl.col('cum_size') <= level) | 
                (pl.col('cum_size').shift(1).over('timeMs').fill_null(0) < level)
            )
            .with_columns([
                # Clip the last size to not exceed level
                pl.when(pl.col('cum_size') > level)
                  .then(pl.col('size') - (pl.col('cum_size') - level))
                  .otherwise(pl.col('size'))
                  .alias('clipped_size')
            ])
            .group_by('timeMs')
            .agg([
                (pl.col('fwd') * pl.col('clipped_size')).sum().alias('weighted_sum'),
                pl.col('clipped_size').sum().alias('total_size'),
            ])
            .with_columns([
                (pl.col('weighted_sum') / pl.col('total_size')).alias(f'best_sym_long_{int(level)}')
            ])
            .select(['timeMs', f'best_sym_long_{int(level)}'])
        )
        
        # SHORT: sort by fwd (descending), take cumulative sum up to level
        short_agg = (
            short_opps
            .sort(['timeMs', 'fwd'], descending=[False, True])
            .with_columns([
                pl.col('size').cum_sum().over('timeMs').alias('cum_size'),
            ])
            .filter(
                (pl.col('cum_size') <= level) | 
                (pl.col('cum_size').shift(1).over('timeMs').fill_null(0) < level)
            )
            .with_columns([
                pl.when(pl.col('cum_size') > level)
                  .then(pl.col('size') - (pl.col('cum_size') - level))
                  .otherwise(pl.col('size'))
                  .alias('clipped_size')
            ])
            .group_by('timeMs')
            .agg([
                (pl.col('fwd') * pl.col('clipped_size')).sum().alias('weighted_sum'),
                pl.col('clipped_size').sum().alias('total_size'),
            ])
            .with_columns([
                (pl.col('weighted_sum') / pl.col('total_size')).alias(f'best_sym_short_{int(level)}')
            ])
            .select(['timeMs', f'best_sym_short_{int(level)}'])
        )
        
        level_results.append((long_agg, short_agg))
    
    # Combine all results
    result = lf.group_by('timeMs').agg(agg_exprs)
    
    for long_agg, short_agg in level_results:
        result = result.join(long_agg, on='timeMs', how='left')
        result = result.join(short_agg, on='timeMs', how='left')
    
    result = result.sort('timeMs')
    
    if verbose:
        print(f"  Aggregated for {len(levels)} contract levels: {levels}")
    
    return result

# =============================================================================
# Alternative Approach: Arbitrage using only synthetic futures contracts
# =============================================================================

def calculate_synthetics(
    store,
    inst_family: str,
    dates: list[date],
    binning: str,
    assets: float = 0,
    volume: float = 0,
    verbose: bool = False,
    batch_days: int = 5,
    max_quote_age_ms: int = 60 * 1000,
) -> pl.LazyFrame:
    if verbose:
        start_time = datetime.now()
        print(f"Calculating put-call parity arbitrage opportunities for {inst_family} for {len(dates)} dates using {binning} binning")

    # =============================================================================
    # Step 1: Fetch and preprocess options and spot data
    # =============================================================================

    # Prepare shared parameters
    shared_params = {
        'inst_family': inst_family,
        'dates': dates,
        'binning': binning,
        'verbose': verbose,
        'batch_days': batch_days,
        'cache_name': f'{binning}_market_snapshots'
    }

    # Fetch options data
    options_features = ['trim', 'bin_ff', 'nullify', 'drop_nulls_strict', 'parse_option', 'tenor']
    options_lf = store.get(
        inst_type='OPTION',
        depth=1,
        **shared_params,
        features=options_features,
    ).filter((pl.col('time_bin') - pl.col('timeMs')) <= max_quote_age_ms)
    options_lf = sink_bins(options_lf)
    # Pair options
    options_lf = pair_options(options_lf, include_qty=True)

    # Fetch spot data
    spot_features = ['trim', 'strip', 'bin_ff', 'sink_bins']
    spot_lf = store.get(
        inst_type='SPOT',
        depth=0,
        **shared_params,
        features=spot_features,
    ).drop('symbol').rename({'mid': 'BTC-USD'})

    # Join spot to add BTC-USD column
    lf = options_lf.join(spot_lf, on=['timeMs'], how='left')

    # Join and add moneyness - using spot price for moneyness since we're not comparing against listed futures.
    lf = lf.with_columns((pl.col('strike') / pl.col('BTC-USD')).log().alias('moneyness'))

    if verbose:
        time_1 = datetime.now()
        print(f" - Time taken to fetch options and spot data: {time_1 - start_time}")

    # =============================================================================
    # Step 2: Calculate parity arbitrage
    # =============================================================================
    
    # Get fee rates
    options_tier, options_maker_fee, options_taker_fee = get_options_fees(assets, volume)
    
    if verbose:
        print(f"  {options_tier:>8} options fees: maker={options_maker_fee:.4%}, taker={options_taker_fee:.4%}")
    
    # Define synthetic order types: (name, direction, call_order, put_order, description)
    # Synthetics where both legs are limit orders are not included as they do not represent executable opportunities.
    order_defs = [
        # LONG synthetics (buy call, sell put)
        # ('LONG_LMT_Cb_Pa', 'long', 'maker', 'maker', 'Long limit: buy call @ bid, sell put @ ask'),
        ('LONG_SYM_Cb_Pb', 'long', 'maker', 'taker', 'Long sym: buy call @ bid, sell put @ bid'),
        ('LONG_SYM_Ca_Pa', 'long', 'taker', 'maker', 'Long sym: buy call @ ask, sell put @ ask'),
        ('LONG_MKT_Ca_Pb', 'long', 'taker', 'taker', 'Long market: buy call @ ask, sell put @ bid'),
        # SHORT synthetics (sell call, buy put)
        # ('SHORT_LMT_Ca_Pb', 'short', 'maker', 'maker', 'Short limit: sell call @ ask, buy put @ bid'),
        ('SHORT_SYM_Ca_Pa', 'short', 'maker', 'taker', 'Short sym: sell call @ ask, buy put @ ask'),
        ('SHORT_SYM_Cb_Pb', 'short', 'taker', 'maker', 'Short sym: sell call @ bid, buy put @ bid'),
        ('SHORT_MKT_Cb_Pa', 'short', 'taker', 'taker', 'Short market: sell call @ bid, buy put @ ask'),
    ]
    
    # Calculate synthetic forwards and arbitrage for each order type
    columns = []
    
    option_tick = 0.0001
    contract_multiplier = 0.01 if inst_family == 'BTC-USD' else 0.1
    inverse_multiplier = 100 if inst_family == 'BTC-USD' else 10
    # If order is maker, we must add/subtract one tick to be at the top of the order book.
    # If order is taker, we are limited in size by the qty of the top bid/ask.
    # If the order is maker, we take qty to be inverse_multiplier contracts.
    for name, direction, call_order, put_order, description in order_defs:
        # Select prices based on order type
        if direction == 'long':
            call_px = (pl.col('call_bid') + option_tick) if call_order == 'maker' else pl.col('call_ask') # Buy call
            put_px = (pl.col('put_ask') - option_tick) if put_order == 'maker' else pl.col('put_bid') # Sell put
            call_qty = inverse_multiplier if call_order == 'maker' else pl.col('call_ask_qty')
            put_qty = inverse_multiplier if put_order == 'maker' else pl.col('put_bid_qty')
            call_fee_rate = options_maker_fee if call_order == 'maker' else options_taker_fee
            put_fee_rate = options_maker_fee if put_order == 'maker' else options_taker_fee
        else:  # short
            call_px = (pl.col('call_ask') - option_tick) if call_order == 'maker' else pl.col('call_bid') # Sell call
            put_px = (pl.col('put_bid') + option_tick) if put_order == 'maker' else pl.col('put_ask') # Buy put
            call_qty = inverse_multiplier if call_order == 'maker' else pl.col('call_bid_qty')
            put_qty = inverse_multiplier if put_order == 'maker' else pl.col('put_ask_qty')
            call_fee_rate = options_maker_fee if call_order == 'maker' else options_taker_fee
            put_fee_rate = options_maker_fee if put_order == 'maker' else options_taker_fee
        
        # Convert prices to USD
        call_amt = call_px * pl.col('BTC-USD')
        put_amt = put_px * pl.col('BTC-USD')
        
        # Position size (limited by option liquidity, in units of contracts)
        position_size = pl.min_horizontal(call_qty, put_qty)
        
        # As confirmed by sanity check, the quotes prices are for one BTC worth of contracts.
        # This means that the premium is for 100 contracts (for BTC-USD) or 10 contracts (for ETH-USD).

        # Calculate net premium
        if direction == 'long':
            net_premium = (call_amt - put_amt)
        else:
            net_premium = (put_amt - call_amt)

        # Calculate fees
        net_fees = (call_amt * call_fee_rate + put_amt * put_fee_rate)
        net_capital = net_premium + net_fees
        
        # Calculated quoted synthetic forwards prices and profit (in USD, including both premium and fees).
        if direction == 'long':
            # F_synthetic = K + (C - P) = K + net_premium
            # Including fees: F_synthetic = K + net_premium + net_fees
            synth_fwd = pl.col('strike') + net_capital
        else:
            # F_synthetic = K - (P - C) = K - net_premium
            # Including fees: F_synthetic = K - net_premium - net_fees
            synth_fwd = pl.col('strike') - net_capital
        
        # Add columns
        columns.extend([
            synth_fwd.alias(f'{name}_fwd'),
            position_size.alias(f'{name}_size'),
            net_capital.alias(f'{name}_capital'),
        ])
    
    lf = lf.with_columns(columns)
    
    if verbose:
        end_time = datetime.now()
        print(f" - Time taken to calculate parity arbitrage: {end_time - time_1}")
        print(f" - Total time taken for parity arbitrage: {end_time - start_time}")

    return lf

def aggregate_synthetics(
    lf: pl.LazyFrame,
    inst_family: str,
    levels: list[int] = [1],
    max_return: float = 10.0,
    verbose: bool = False,
) -> pl.LazyFrame:
    """
    Aggregate synthetic forward arbitrage opportunities by finding the best
    market synthetic long and short for each (timeMs, expiry) combination.
    
    For each time-expiry pair:
    - Best long: lowest LONG_MKT_Ca_Pb_fwd (buy synthetic forward cheap)
    - Best short: highest SHORT_MKT_Cb_Pa_fwd (sell synthetic forward expensive)
    - For SYM orders: aggregate top N opportunities (by levels) using size-weighted mean
    - Calculate arbitrage profit, return, and annualized return
    
    Args:
        lf: LazyFrame with synthetic forward prices
        inst_family: Instrument family (e.g., 'BTC-USD')
        levels: List of level sizes for SYM aggregation (e.g., [1, 5])
        max_return: Maximum return to clip outliers
        verbose: Whether to print progress
    """
    if verbose:
        print(f"Aggregating synthetic forward arbitrage opportunities for levels {levels}...")
    
    # Get contract multiplier
    contract_multiplier = 0.01 if inst_family == 'BTC-USD' else 0.1
    
    # Find best long synthetic (minimum forward price) for each (timeMs, expiry)
    best_long = (
        lf
        .group_by(['timeMs', 'expiry'])
        .agg([
            pl.col('T').filter(pl.col('LONG_MKT_Ca_Pb_fwd') == pl.col('LONG_MKT_Ca_Pb_fwd').min()).first().alias('T'),
            pl.col('BTC-USD').filter(pl.col('LONG_MKT_Ca_Pb_fwd') == pl.col('LONG_MKT_Ca_Pb_fwd').min()).first().alias('BTC-USD'),
            pl.col('LONG_MKT_Ca_Pb_fwd').min().alias('long_fwd'),
            pl.col('strike').filter(pl.col('LONG_MKT_Ca_Pb_fwd') == pl.col('LONG_MKT_Ca_Pb_fwd').min()).first().alias('long_strike'),
            pl.col('LONG_MKT_Ca_Pb_size').filter(pl.col('LONG_MKT_Ca_Pb_fwd') == pl.col('LONG_MKT_Ca_Pb_fwd').min()).first().alias('long_size'),
            pl.col('LONG_MKT_Ca_Pb_capital').filter(pl.col('LONG_MKT_Ca_Pb_fwd') == pl.col('LONG_MKT_Ca_Pb_fwd').min()).first().alias('long_capital'),
            pl.col('call_ask').filter(pl.col('LONG_MKT_Ca_Pb_fwd') == pl.col('LONG_MKT_Ca_Pb_fwd').min()).first().alias('long_call_ask'),
            pl.col('put_bid').filter(pl.col('LONG_MKT_Ca_Pb_fwd') == pl.col('LONG_MKT_Ca_Pb_fwd').min()).first().alias('long_put_bid'),
        ])
    )
    
    # Find best short synthetic (maximum forward price) for each (timeMs, expiry)
    best_short = (
        lf
        .group_by(['timeMs', 'expiry'])
        .agg([
            pl.col('SHORT_MKT_Cb_Pa_fwd').max().alias('short_fwd'),
            pl.col('strike').filter(pl.col('SHORT_MKT_Cb_Pa_fwd') == pl.col('SHORT_MKT_Cb_Pa_fwd').max()).first().alias('short_strike'),
            pl.col('SHORT_MKT_Cb_Pa_size').filter(pl.col('SHORT_MKT_Cb_Pa_fwd') == pl.col('SHORT_MKT_Cb_Pa_fwd').max()).first().alias('short_size'),
            pl.col('SHORT_MKT_Cb_Pa_capital').filter(pl.col('SHORT_MKT_Cb_Pa_fwd') == pl.col('SHORT_MKT_Cb_Pa_fwd').max()).first().alias('short_capital'),
            pl.col('call_bid').filter(pl.col('SHORT_MKT_Cb_Pa_fwd') == pl.col('SHORT_MKT_Cb_Pa_fwd').max()).first().alias('short_call_bid'),
            pl.col('put_ask').filter(pl.col('SHORT_MKT_Cb_Pa_fwd') == pl.col('SHORT_MKT_Cb_Pa_fwd').max()).first().alias('short_put_ask'),
        ])
    )
    
    # Join best long and short
    result_lf = best_long.join(best_short, on=['timeMs', 'expiry'], how='inner')
    
    # =============================================================================
    # Aggregate SYM synthetics for each level
    # =============================================================================
    
    for level in levels:
        if verbose:
            print(f"  Processing SYM level {level}...")
        
        # Prepare LONG SYM opportunities (combine both types)
        long_sym_opps = lf.select([
            'timeMs', 'expiry', 'strike',
            pl.col('LONG_SYM_Cb_Pb_fwd').alias('fwd'),
            pl.col('LONG_SYM_Cb_Pb_size').alias('size'),
            pl.col('LONG_SYM_Cb_Pb_capital').alias('capital'),
            pl.lit('Cb_Pb').alias('type'),
        ]).vstack(
            lf.select([
                'timeMs', 'expiry', 'strike',
                pl.col('LONG_SYM_Ca_Pa_fwd').alias('fwd'),
                pl.col('LONG_SYM_Ca_Pa_size').alias('size'),
                pl.col('LONG_SYM_Ca_Pa_capital').alias('capital'),
                pl.lit('Ca_Pa').alias('type'),
            ])
        )
        
        # Aggregate top N long opportunities by lowest forward price
        long_sym_agg = (
            long_sym_opps
            .sort(['timeMs', 'expiry', 'fwd'])
            .with_columns([
                pl.col('fwd').rank(method='ordinal').over(['timeMs', 'expiry']).alias('rank'),
            ])
            .filter(pl.col('rank') <= level)
            .group_by(['timeMs', 'expiry'])
            .agg([
                # Size-weighted mean of forward price
                (pl.col('fwd') * pl.col('size')).sum().alias('weighted_fwd_sum'),
                pl.col('size').sum().alias('total_size'),
                # Sum of capital requirements
                (pl.col('capital') * pl.col('size')).sum().alias('weighted_capital_sum'),
            ])
            .with_columns([
                (pl.col('weighted_fwd_sum') / pl.col('total_size')).alias(f'sym_long_fwd_{level}'),
                pl.col('total_size').alias(f'sym_long_size_{level}'),
                (pl.col('weighted_capital_sum') / pl.col('total_size')).alias(f'sym_long_capital_{level}'),
            ])
            .select(['timeMs', 'expiry', f'sym_long_fwd_{level}', f'sym_long_size_{level}', f'sym_long_capital_{level}'])
        )
        
        # Prepare SHORT SYM opportunities (combine both types)
        short_sym_opps = lf.select([
            'timeMs', 'expiry', 'strike',
            pl.col('SHORT_SYM_Ca_Pa_fwd').alias('fwd'),
            pl.col('SHORT_SYM_Ca_Pa_size').alias('size'),
            pl.col('SHORT_SYM_Ca_Pa_capital').alias('capital'),
            pl.lit('Ca_Pa').alias('type'),
        ]).vstack(
            lf.select([
                'timeMs', 'expiry', 'strike',
                pl.col('SHORT_SYM_Cb_Pb_fwd').alias('fwd'),
                pl.col('SHORT_SYM_Cb_Pb_size').alias('size'),
                pl.col('SHORT_SYM_Cb_Pb_capital').alias('capital'),
                pl.lit('Cb_Pb').alias('type'),
            ])
        )
        
        # Aggregate top N short opportunities by highest forward price
        short_sym_agg = (
            short_sym_opps
            .sort(['timeMs', 'expiry', 'fwd'], descending=[False, False, True])
            .with_columns([
                pl.col('fwd').rank(method='ordinal', descending=True).over(['timeMs', 'expiry']).alias('rank'),
            ])
            .filter(pl.col('rank') <= level)
            .group_by(['timeMs', 'expiry'])
            .agg([
                # Size-weighted mean of forward price
                (pl.col('fwd') * pl.col('size')).sum().alias('weighted_fwd_sum'),
                pl.col('size').sum().alias('total_size'),
                # Sum of capital requirements
                (pl.col('capital') * pl.col('size')).sum().alias('weighted_capital_sum'),
            ])
            .with_columns([
                (pl.col('weighted_fwd_sum') / pl.col('total_size')).alias(f'sym_short_fwd_{level}'),
                pl.col('total_size').alias(f'sym_short_size_{level}'),
                (pl.col('weighted_capital_sum') / pl.col('total_size')).alias(f'sym_short_capital_{level}'),
            ])
            .select(['timeMs', 'expiry', f'sym_short_fwd_{level}', f'sym_short_size_{level}', f'sym_short_capital_{level}'])
        )
        
        # Join SYM aggregations
        result_lf = result_lf.join(long_sym_agg, on=['timeMs', 'expiry'], how='left')
        result_lf = result_lf.join(short_sym_agg, on=['timeMs', 'expiry'], how='left')
        
        # Calculate position size, capital, profit, and returns for this level
        result_lf = result_lf.with_columns([
            # Position size (limited by minimum of long and short sizes)
            pl.min_horizontal(f'sym_long_size_{level}', f'sym_short_size_{level}').alias(f'sym_position_size_{level}'),
        ])
        
        result_lf = result_lf.with_columns([
            # Profit per unit (sell high - buy low)
            (pl.col(f'sym_short_fwd_{level}') - pl.col(f'sym_long_fwd_{level}')).alias(f'sym_profit_per_unit_{level}'),
            
            # Capital per unit (sum of long and short capital requirements)
            (pl.col(f'sym_long_capital_{level}') + pl.col(f'sym_short_capital_{level}')).alias(f'sym_capital_per_unit_{level}'),
        ])
        
        # Scale by position size and contract multiplier
        result_lf = result_lf.with_columns([
            (pl.col(f'sym_profit_per_unit_{level}') * contract_multiplier * pl.col(f'sym_position_size_{level}')).alias(f'sym_profit_{level}'),
            (pl.col(f'sym_capital_per_unit_{level}') * contract_multiplier * pl.col(f'sym_position_size_{level}')).alias(f'sym_capital_{level}'),
        ])
        
        # Calculate returns
        result_lf = result_lf.with_columns([
            # Raw return (NaN when capital <= 0)
            pl.when(pl.col(f'sym_capital_{level}') > 0)
              .then(pl.col(f'sym_profit_{level}') / pl.col(f'sym_capital_{level}'))
              .otherwise(pl.lit(float('nan')))
              .alias(f'sym_raw_return_{level}'),
            
            # Profitable flag
            (pl.col(f'sym_profit_{level}') > 0).alias(f'sym_profitable_{level}'),
        ])
        
        # Clip returns
        result_lf = result_lf.with_columns([
            pl.col(f'sym_raw_return_{level}').clip(upper_bound=max_return).alias(f'sym_return_{level}'),
        ])
        
        # Calculate annualized return
        result_lf = result_lf.with_columns([
            pl.when(pl.col(f'sym_return_{level}').is_not_nan())
              .then((1 + pl.col(f'sym_return_{level}')).pow(1 / pl.col('T')) - 1)
              .otherwise(pl.lit(float('nan')))
              .alias(f'sym_annualized_return_{level}'),
        ])
    
    # =============================================================================
    # Calculate MKT arbitrage metrics
    # =============================================================================
    
    # Calculate position size (limited by minimum of long and short sizes)
    result_lf = result_lf.with_columns([
        pl.min_horizontal('long_size', 'short_size').alias('position_size'),
    ])
    
    # Calculate total capital and profit, accounting for contract multiplier and position size
    result_lf = result_lf.with_columns([
        # Profit per unit (sell high - buy low)
        (pl.col('short_fwd') - pl.col('long_fwd')).alias('profit_per_unit'),
        
        # Capital per unit (sum of long and short capital requirements)
        (pl.col('long_capital') + pl.col('short_capital')).alias('capital_per_unit'),
    ])
    
    # Scale by position size and contract multiplier
    result_lf = result_lf.with_columns([
        (pl.col('profit_per_unit') * contract_multiplier * pl.col('position_size')).alias('profit'),
        (pl.col('capital_per_unit') * contract_multiplier * pl.col('position_size')).alias('capital'),
    ])
    
    # Calculate returns (handle negative capital case)
    result_lf = result_lf.with_columns([
        # Raw return (NaN when capital <= 0)
        pl.when(pl.col('capital') > 0)
          .then(pl.col('profit') / pl.col('capital'))
          .otherwise(pl.lit(float('nan')))
          .alias('raw_return'),
        
        # Profitable flag
        (pl.col('profit') > 0).alias('profitable'),
    ])
    
    # Clip returns (only upper bound since best selection should ensure positive returns)
    result_lf = result_lf.with_columns([
        pl.col('raw_return').clip(upper_bound=max_return).alias('return'),
    ])
    
    # Calculate annualized return using compound formula
    result_lf = result_lf.with_columns([
        pl.when(pl.col('return').is_not_nan())
          .then((1 + pl.col('return')).pow(1 / pl.col('T')) - 1)
          .otherwise(pl.lit(float('nan')))
          .alias('annualized_return'),
    ])
    
    # Sort by expiry and timeMs
    result_lf = result_lf.sort(['expiry', 'timeMs'])
    
    if verbose:
        print("Aggregation complete.")
    
    return result_lf