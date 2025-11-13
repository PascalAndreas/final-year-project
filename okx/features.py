import polars as pl
from okx.helpers import parse_option_name, parse_future_name
from datetime import datetime
from typing import Optional, Callable

# =============================================================================
# Feature registry (expressions and flush features)
# =============================================================================

# Registry features are simple Polars expressions that can be batched together.
FEATURES = {
    'mid': (pl.col('ask_1_px') + pl.col('bid_1_px')) / 2,
    'spread': pl.col('ask_1_px') - pl.col('bid_1_px'),
    'rel_spread': (pl.col('ask_1_px') - pl.col('bid_1_px')) / ((pl.col('ask_1_px') + pl.col('bid_1_px')) / 2),
    'imbalance1': (pl.col('bid_1_qty') - pl.col('ask_1_qty')) / (pl.col('bid_1_qty') + pl.col('ask_1_qty')),
    'imbalance5': (
        (sum(pl.col(f'bid_{i}_qty') for i in range(1, 6)) - sum(pl.col(f'ask_{i}_qty') for i in range(1, 6)))
        / (sum(pl.col(f'bid_{i}_qty') for i in range(1, 6)) + sum(pl.col(f'ask_{i}_qty') for i in range(1, 6)))
    ),
    'bid_volume': sum(pl.col(f'bid_{i}_qty') for i in range(1, 6)),
    'ask_volume': sum(pl.col(f'ask_{i}_qty') for i in range(1, 6)),
}

# Transformations that must be executed immediately (cannot sit in the FEATURES expr bucket)
# because they either depend on runtime config (depth/binning), mutate the schema, or require
# maintaining ordering/uniqueness guarantees.
def build_flush_features(depth: Optional[int], binning: Optional[str], inst_type: str) -> dict[str, Callable]:
    return {
        'trim': (lambda lf: (trim_ob(lf, depth), True)) if depth is not None else (lambda lf: (lf, False)),
        'bin': (lambda lf: (bin_ob(lf, binning), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_count': (lambda lf: (bin_ob(lf, binning, count=True), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_ff': (lambda lf: (bin_ob(lf, binning, ff=True), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_count_ff': (lambda lf: (bin_ob(lf, binning, count=True, ff=True), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_ff_count': (lambda lf: (bin_ob(lf, binning, count=True, ff=True), True)) if binning is not None else (lambda lf: (lf, False)),
        'tenor': lambda lf: (add_tenor(lf, inst_type), True),
        'log': lambda lf: (log_prices(lf), True),
        'exp': lambda lf: (exp_prices(lf), True),
        'strip': lambda lf: (strip_ob(lf), True),
        'parse_option': lambda lf: (parse_option(lf), True),
        'nullify': lambda lf: (nullify(lf), True),
        'drop_nulls': lambda lf: (drop_nulls(lf), True),
        'dedupe': lambda lf: (lf.unique(maintain_order=True), True),
    }

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
    Keep only minimal columns: timeMs/time_bin, symbol, top-level price cols (bid_1_px, ask_1_px, mid, spread, rel_spread).
    Log-space prices are kept if present.
    """
    cols = lf.collect_schema().names()
    keep_cols = ['timeMs', 'symbol']
    keep_if_present = [
        'time_bin', 'ln_bid_1_px', 'ln_ask_1_px',
        'bid_1_px', 'ask_1_px', 'ln_mid', 'mid', 'spread', 'rel_spread'
    ]
    keep_cols.extend([col for col in keep_if_present if col in cols])
    return lf.select(keep_cols)

def bin_ob(lf: pl.LazyFrame, freq: str, count: bool = False, ff: bool = False) -> pl.LazyFrame:
    """
    Bin orderbook timestamps and keep most recent entry per time bin per symbol.
    
    Args:
        freq: Polars duration string (e.g., '5m', '1m', '30s', '100ms')
        count: Add 'bin_count' column with number of entries in each bin
        ff: Forward fill to create complete (symbol, time_bin) grid
    
    Returns:
        LazyFrame with 'time_bin' column added (and 'bin_count' if count=True)
    """
    # Bin the data
    agg_exprs = [pl.col('*').last(), pl.len().alias('bin_count')] if count else [pl.all().last()]
    
    binned = (
        lf
        .with_columns(pl.col('timeMs').cast(pl.Datetime('ms')).alias('_ts'))
        .sort(['symbol', '_ts'])
        .group_by_dynamic('_ts', every=freq, group_by='symbol', closed='right', label='right')
        .agg(agg_exprs)
        .with_columns(pl.col('_ts').alias('time_bin'))
        .drop('_ts')
    )
    
    # Forward fill if requested
    if ff:
        # Create per-symbol grids based on each symbol's actual trading range
        # This avoids creating bins before a symbol started trading
        symbol_ranges = (
            binned
            .group_by('symbol')
            .agg([
                pl.col('time_bin').min().alias('min_time'),
                pl.col('time_bin').max().alias('max_time'),
            ])
        )
        
        # Get all unique time bins
        all_time_bins = binned.select('time_bin').unique().sort('time_bin')
        
        # For each symbol, create grid only within its trading range
        complete_grid = (
            symbol_ranges
            .join(all_time_bins, how='cross')
            .filter(
                (pl.col('time_bin') >= pl.col('min_time')) & 
                (pl.col('time_bin') <= pl.col('max_time'))
            )
            .select(['symbol', 'time_bin'])
        )
        
        # Join and forward fill
        binned = (
            complete_grid
            .sort(['symbol', 'time_bin'])
            .join(binned, on=['symbol', 'time_bin'], how='left')
            .with_columns(
                pl.all().exclude(['symbol', 'time_bin'] + (['bin_count'] if count else [])).forward_fill().over('symbol')
            )
        )
        
        # Fill missing bin_count with 0 if count=True
        if count:
            binned = binned.with_columns(pl.col('bin_count').fill_null(0))
    
    return binned

# ===============================================================
# Price transformation functions
# ===============================================================

def log_prices(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Transform _px columns to ln_px (log-space), dropping originals."""
    cols = lf.collect_schema().names()
    price_cols = [col for col in cols if col.endswith('_px') or col in ['mid', 'spread']]
    
    if not price_cols:
        return lf
    
    log_exprs = [pl.col(col).log().alias(f'ln_{col}') for col in price_cols]
    return lf.with_columns(log_exprs).drop(price_cols)

def exp_prices(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Transform ln_px columns back to px (exp), dropping originals."""
    cols = lf.collect_schema().names()
    log_cols = [col for col in cols if col.startswith('ln_')]
    
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

def drop_nulls(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filter out rows where both bid and ask are null.
    
    Useful after bin_ff which can create null rows when forward-filling
    before a symbol's first datapoint or during data gaps.
    """
    # Check for log-space columns first (ln_bid_1_px/ln_ask_1_px)
    cols = lf.collect_schema().names()
    if 'ln_bid_1_px' in cols and 'ln_ask_1_px' in cols:
        return lf.filter(
            pl.col('ln_bid_1_px').is_not_null() | pl.col('ln_ask_1_px').is_not_null()
        )
    elif 'bid_1_px' in cols and 'ask_1_px' in cols:
        return lf.filter(
            pl.col('bid_1_px').is_not_null() | pl.col('ask_1_px').is_not_null()
        )
    else:
        return lf

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
    For SWAP/SPOT: sets expiry = timeMs (making T = 0)
    
    Symbol formats:
    - FUTURES: BTC-USD-250131 (date is 3rd element)
    - OPTION: BTC-USD-250627-200000-C (date is 3rd element)
    """
    # Parse expiry based on instrument type (as milliseconds)
    # Using vectorized string operations - much faster than map_elements
    if inst_type in ['SWAP', 'SPOT']:
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