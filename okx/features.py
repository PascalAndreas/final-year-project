import polars as pl
from okx.helpers import parse_option_name, parse_future_name
from datetime import datetime

# ===============================================================
# Basic orderbook processing functions
# ===============================================================

def trim_ob(lf: pl.LazyFrame, n_levels: int = 5) -> pl.LazyFrame:
    """
    Trim orderbook LazyFrame to keep only top n levels.
    """
    if n_levels == 0:
        # Calculate mid price and drop all orderbook columns
        return lf.with_columns([
            ((pl.col('bid_1_px') + pl.col('ask_1_px')) / 2).alias('mid')
        ]).select([
            col for col in lf.collect_schema().names()
            if not any(col.startswith(f'{side}_') for side in ['ask', 'bid'])
        ] + ['mid'])
    
    # Trim to n levels
    cols_to_keep = [
        col for col in lf.collect_schema().names()
        if not any(col.startswith(f'{side}_') for side in ['ask', 'bid']) or
        (col.split('_')[1].isdigit() and int(col.split('_')[1]) <= n_levels)
    ]
    return lf.select(cols_to_keep)


def strip_ob(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Strip orderbook to minimal columns: timeMs, symbol, time_bin (if present), and top-level prices.
    
    Works with both regular and log-space prices:
    - Regular: keeps bid_1_px, ask_1_px (or mid if present)
    - Log-space: keeps ln_bid_1_px, ln_ask_1_px (or ln_mid if present)
    
    Returns:
        LazyFrame with only [timeMs, symbol, (time_bin), price columns]
    """
    schema = lf.collect_schema()
    cols = schema.names()
    
    # Base columns to always keep, include time_bin if present
    keep_cols = ['timeMs', 'symbol']
    if 'time_bin' in cols:
        keep_cols.append('time_bin')
    
    # Check for log-space vs regular prices
    if 'ln_bid_1_px' in cols and 'ln_ask_1_px' in cols:
        keep_cols.extend(['ln_bid_1_px', 'ln_ask_1_px'])
    elif 'bid_1_px' in cols and 'ask_1_px' in cols:
        keep_cols.extend(['bid_1_px', 'ask_1_px'])
    
    # Add mid if present (either log or regular)
    if 'ln_mid' in cols:
        keep_cols.append('ln_mid')
    elif 'mid' in cols:
        keep_cols.append('mid')
    
    return lf.select([col for col in keep_cols if col in cols])

def bin_ob(lf: pl.LazyFrame, freq: str) -> pl.LazyFrame:
    """
    Bin orderbook timestamps and keep most recent entry per time bin per symbol.
    Adds a 'time_bin' column with the bin timestamp. Leaves timeMs and exchTimeMs unchanged.
    freq: Polars duration string (e.g., '5m', '1m', '30s', '100ms')
    """
    # Convert timeMs to datetime for grouping
    return (lf
        .with_columns(pl.col('timeMs').cast(pl.Datetime('ms')).alias('_ts'))
        .sort('_ts')
        .group_by_dynamic('_ts', every=freq, by='symbol', closed='right', label='right')
        .agg([pl.all().last()])
        .with_columns(
            pl.col('_ts').alias('time_bin')  # Keep as datetime
        )
        .drop('_ts')
        .sort(['symbol', 'timeMs'])
    )

# ===============================================================
# Price transformation functions
# ===============================================================

def log_prices(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Transform _px columns to ln_px (log-space), dropping originals."""
    schema = lf.collect_schema()
    price_cols = [col for col in schema.names() if col.endswith('_px') or col in ['mid', 'spread']]
    
    if not price_cols:
        return lf
    
    log_exprs = [pl.col(col).log().alias(f'ln_{col}') for col in price_cols]
    return lf.with_columns(log_exprs).drop(price_cols)


def exp_prices(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Transform ln_px columns back to px (exp), dropping originals."""
    schema = lf.collect_schema()
    log_cols = [col for col in schema.names() if col.startswith('ln_')]
    
    if not log_cols:
        return lf
    
    exp_exprs = [pl.col(col).exp().alias(col[3:]) for col in log_cols]
    return lf.with_columns(exp_exprs).drop(log_cols)

def nullify(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Replace zero prices with null for bid/ask columns."""
    return lf.with_columns([
        pl.when(pl.col('bid_1_px') > 0).then(pl.col('bid_1_px')).otherwise(None).alias('bid_1_px'),
        pl.when(pl.col('ask_1_px') > 0).then(pl.col('ask_1_px')).otherwise(None).alias('ask_1_px'),
    ])


# ===============================================================
# Symbol parsing features
# ===============================================================

def parse_option(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Parse option symbols to add strike and opt_type columns.
    
    Adds:
        - strike: Strike price (int)
        - opt_type: 'C' for call, 'P' for put
    
    Note: Does NOT add moneyness (requires spot reference from separate data source).
    
    Symbol format: BTC-USD-250627-200000-C (currency-date-strike-type)
    """
    # Remove file extension if present (e.g., '.OK') and split by '-'
    # Using vectorized string operations - much faster than map_elements
    return lf.with_columns([
        # Extract strike (4th element after split, index 3)
        pl.col('symbol')
          .str.split('.')
          .list.first()  # Remove extension
          .str.split('-')
          .list.get(3)
          .cast(pl.Int64)
          .alias('strike'),
        # Extract option type (5th element after split, index 4)
        pl.col('symbol')
          .str.split('.')
          .list.first()
          .str.split('-')
          .list.get(4)
          .str.to_uppercase()
          .alias('opt_type'),
    ])




def add_tenor(lf: pl.LazyFrame, inst_type: str) -> pl.LazyFrame:
    """
    Add expiry (ms) and T (time to maturity in years) by parsing symbols.
    
    For FUTURES/OPTION: parses expiry from symbol
    For SWAP: sets expiry = timeMs (making T = 0)
    
    Symbol formats:
    - FUTURES: BTC-USD-250131 (date is 3rd element)
    - OPTION: BTC-USD-250627-200000-C (date is 3rd element)
    """
    # Parse expiry based on instrument type (as milliseconds)
    # Using vectorized string operations - much faster than map_elements
    if inst_type == 'SWAP':
        # Set expiry = timeMs so that T = 0 for perpetual swaps
        expiry_ms = pl.col('timeMs')
    elif inst_type in ['FUTURES', 'OPTION']:
        # Extract date string (3rd element, index 2) and parse
        # Format is YYMMDD, expiry at 08:00 UTC
        date_str = (pl.col('symbol')
                    .str.split('.')
                    .list.first()  # Remove extension if present
                    .str.split('-')
                    .list.get(2))  # Get date part (YYMMDD)
        
        # Parse YYMMDD to datetime at 08:00 UTC, then convert to milliseconds
        # strptime format: %y%m%d
        expiry_ms = (date_str
                     .str.strptime(pl.Datetime('us'), '%y%m%d')
                     .dt.replace_time_zone('UTC')
                     .dt.offset_by('8h')  # Add 8 hours for 08:00 UTC expiry
                     .dt.timestamp('ms'))
    else:
        # Unknown instrument type, return unchanged
        return lf
    
    # Add expiry (ms) and T columns. ACT/365F convention.
    return lf.with_columns([
        expiry_ms.alias('expiry'),
        ((expiry_ms - pl.col('timeMs')) / 1000.0 / (365 * 24 * 3600)).alias('T')
    ])