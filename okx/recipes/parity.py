import polars as pl
from datetime import date, datetime
from typing import Optional
from okx.recipes.helpers import pair_options
from okx.fees import get_fees, get_options_fees

def prepare_market_snapshots(
    store,
    inst_family: str,
    dates: list[date],
    binning: str,
    min_time_to_expiry_hours: float = 2.0,
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
    options_features = ['trim', 'nullify', 'bin_ff', 'sink_bins', 'drop_nulls_strict', 'parse_option', 'tenor']
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

def calculate_parity_arbitrage(
    lf: pl.LazyFrame,
    assets: float = 0,
    volume: float = 0,
    max_return: float = 10.0,
    verbose: bool = False,
) -> pl.LazyFrame:
    """
    Calculate put-call parity arbitrage opportunities from market snapshots.
    
    NOTE: Assets and Volume bounds for options fee table are in USD, while the futures fee table bounds are in EUR.
          For now, we're ignoring this discrepancy.
          
    Args:
        lf: LazyFrame from prepare_market_snapshots
        assets: Account assets in USD for options and futures fee tiers
        volume: 30-day trading volume in USD for options and futures fee tiers
        max_return: Maximum return to clip outliers (default: 10.0 = 1000%)
        verbose: Whether to print progress
        
    Returns:
        LazyFrame with synthetic forwards, arbitrage metrics for each order type
    """
    if verbose:
        start_time = datetime.now()
        print("Calculating parity arbitrage opportunities...")
    
    # Get fee rates
    options_tier, options_maker_fee, options_taker_fee = get_options_fees(assets, volume)
    futures_tier, futures_maker_fee, futures_taker_fee = get_fees(assets, volume)
    
    if verbose:
        print(f"  {options_tier:>8} options fees: maker={options_maker_fee:.4%}, taker={options_taker_fee:.4%}")
        print(f"  {futures_tier:>8} futures fees: maker={futures_maker_fee:.4%}, taker={futures_taker_fee:.4%}")
    
    # Convert option prices from BTC to USD
    lf = lf.with_columns([
        (pl.col('call_bid') * pl.col('BTC-USD')).alias('call_bid_usd'),
        (pl.col('call_ask') * pl.col('BTC-USD')).alias('call_ask_usd'),
        (pl.col('put_bid') * pl.col('BTC-USD')).alias('put_bid_usd'),
        (pl.col('put_ask') * pl.col('BTC-USD')).alias('put_ask_usd'),
    ])
    
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
    
    for name, direction, call_order, put_order, description in order_defs:
        # Select prices based on order type
        if direction == 'long':
            call_px = 'call_bid_usd' if call_order == 'maker' else 'call_ask_usd'
            put_px = 'put_ask_usd' if put_order == 'maker' else 'put_bid_usd'
            call_qty = 'call_bid_qty' if call_order == 'maker' else 'call_ask_qty'
            put_qty = 'put_ask_qty' if put_order == 'maker' else 'put_bid_qty'
            call_fee_rate = options_maker_fee if call_order == 'maker' else options_taker_fee
            put_fee_rate = options_maker_fee if put_order == 'maker' else options_taker_fee
        else:  # short
            call_px = 'call_ask_usd' if call_order == 'maker' else 'call_bid_usd'
            put_px = 'put_bid_usd' if put_order == 'maker' else 'put_ask_usd'
            call_qty = 'call_ask_qty' if call_order == 'maker' else 'call_bid_qty'
            put_qty = 'put_bid_qty' if put_order == 'maker' else 'put_ask_qty'
            call_fee_rate = options_maker_fee if call_order == 'maker' else options_taker_fee
            put_fee_rate = options_maker_fee if put_order == 'maker' else options_taker_fee
        
        # Calculate synthetic forward using put-call parity
        call_amt = pl.col(call_px)
        put_amt = pl.col(put_px)
        
        if direction == 'long':
            # F_synthetic = K + (C - P)
            synth_fwd = pl.col('strike') + (call_amt - put_amt)
        else:
            # F_synthetic = K - (P - C) for short
            synth_fwd = pl.col('strike') - (put_amt - call_amt)
        
        # Calculate option fees
        call_fee = call_amt * call_fee_rate
        put_fee = put_amt * put_fee_rate
        option_fees = call_fee + put_fee
        
        # Position size (limited by option liquidity, in BTC)
        position_size = pl.min_horizontal(pl.col(call_qty), pl.col(put_qty))
        
        # Calculate arbitrage metrics
        if direction == 'long':
            # Long synthetic + short futures
            futures_price = pl.col('F_bid')
            futures_fee_rate = futures_taker_fee
            futures_fee = futures_price * futures_fee_rate
            profit_per_contract = futures_price - synth_fwd - option_fees - futures_fee
            net_premium = synth_fwd - pl.col('strike')
            capital_per_contract = net_premium + option_fees + futures_fee
        else:
            # Short synthetic + long futures
            futures_price = pl.col('F_ask')
            futures_fee_rate = futures_taker_fee
            futures_fee = futures_price * futures_fee_rate
            profit_per_contract = synth_fwd - futures_price - option_fees - futures_fee
            net_premium = -(synth_fwd - pl.col('strike'))
            capital_per_contract = net_premium + option_fees + futures_fee
        
        # Total profit and capital
        profit = profit_per_contract * position_size
        capital = capital_per_contract * position_size
        
        # Returns (NaN when capital <= 0 or division issues)
        raw_return = pl.when(capital > 0)
            .then(profit / capital)
            .otherwise(pl.lit(float('nan')))
        
        # Clip returns to prevent extreme outliers
        raw_return_clipped = raw_return.clip(-max_return, max_return)
        
        # Annualized return
        annual_return = pl.when(raw_return_clipped.is_not_nan())
            .then((1 + raw_return_clipped).pow(1 / pl.col('T')) - 1)
            .otherwise(pl.lit(float('nan')))
        
        # Profitable flag
        profitable = profit > 0
        
        # Add columns
        columns.extend([
            synth_fwd.alias(f'{name}_fwd'),
            option_fees.alias(f'{name}_fees'),
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


def summarize_parity_arbitrage(
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