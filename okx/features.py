import polars as pl
from okx.helpers import parse_option_name, parse_future_name
from datetime import datetime
from typing import Optional, Callable
from datetime import datetime, timezone, timedelta
import time

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
def build_flush_features(
    inst_type: str,
    depth: Optional[int], 
    binning: Optional[str], 
    unique_times: Optional[list[int]] = None
) -> dict[str, Callable]:
    """
    Build flush features with runtime configuration.
    
    Args:
        depth: Orderbook depth to trim to
        binning: Frequency string for time-based binning (e.g., '5m')
        inst_type: Instrument type for tenor parsing
        unique_times: Explicit timestamps for binning to specific times
    """
    return {
        # Original binning (uses group_by_dynamic)
        'trim': (lambda lf: (trim_ob(lf, depth), True)) if depth is not None else (lambda lf: (lf, False)),
        'bin': lambda lf: bin_ob(lf, binning, unique_times),
        'bin_count': lambda lf: bin_ob(lf, binning, unique_times, count=True),
        'bin_ff': lambda lf: bin_ob(lf, binning, unique_times, ff=True),
        'bin_count_ff': lambda lf: bin_ob(lf, binning, unique_times, count=True, ff=True),
        'bin_ff_count': lambda lf: bin_ob(lf, binning, unique_times, count=True, ff=True),
        
        # Sink bins (must be done outside batching when using forward fill)
        'sink_bins': lambda lf: (sink_bins(lf), True),
        
        # Other transforms
        'times_to_dt': lambda lf: (times_to_dt(lf), True),
        'times_to_ms': lambda lf: (times_to_ms(lf), True),
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
# Column removal functions
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

# ===============================================================
# Binning functions
# ===============================================================

def _bin(lf: pl.LazyFrame, time_bins: pl.Series, interval: int, count: bool = False, ff: bool = False) -> pl.LazyFrame:
    time_bins_lf = time_bins.to_frame('time_bin').lazy()
    lf = lf.sort('timeMs')
    agg_exprs = [pl.col('*').last(), pl.len().alias('bin_count')] if count else [pl.all().last()]
    binned = (
        lf
        .join_asof(time_bins_lf, left_on='timeMs', right_on='time_bin', strategy='forward')
        .group_by(['symbol', 'time_bin'])
        .agg(agg_exprs).sort(['symbol', 'time_bin'])
    )
    # Step 2: Forward fill if requested (create per-symbol grids based on actual trading ranges)
    if ff:
        # Get per-symbol time ranges
        symbol_ranges = (
            binned
            .group_by('symbol')
            .agg([
                pl.col('time_bin').min().alias('min_time'),
                pl.col('time_bin').max().alias('max_time'),
            ])
        )
        # For each symbol, create grid only within its trading range (plus one interval)
        complete_grid = (
            symbol_ranges
            .join(time_bins_lf, how='cross')
            .filter(
                (pl.col('time_bin') >= pl.col('min_time')) & 
                (pl.col('time_bin') <= pl.col('max_time') + interval)
            )
            .select(['symbol', 'time_bin'])
            .sort(['symbol', 'time_bin'])
        )
        # Join and forward fill
        exclude_cols = ['symbol', 'time_bin']
        if count:
            exclude_cols.append('bin_count')
        
        binned = (
            complete_grid
            .join(binned, on=['symbol', 'time_bin'], how='left', maintain_order='left')
            .with_columns(pl.all().exclude(exclude_cols).forward_fill().over('symbol'))
        )
        # Fill missing bin_count with 0 if count=True
        if count:
            binned = binned.with_columns(pl.col('bin_count').fill_null(0))
    
    return binned

def _make_bins(lf: pl.LazyFrame, freq: str, ff: bool = False) -> tuple[pl.Series, int]:
    time_range = lf.select([
        pl.col('timeMs').min().cast(pl.Datetime('ms')).alias('min_time'),
        pl.col('timeMs').max().cast(pl.Datetime('ms')).alias('max_time')
    ]).collect()
    start_dt = time_range['min_time'][0]
    end_dt = time_range['max_time'][0]
    # Snap to midnight boundaries: start of first day, start of day after last
    start_dt = datetime.combine(start_dt.date(), datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_dt.date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
    
    # Generate bins as datetimes then convert to integer milliseconds
    bins = pl.datetime_range(start_dt, end_dt, interval=freq, closed='right', time_unit='ms', eager=True)
    interval = bins[1] - bins[0]
    if ff:
        # Explicitly specify dtype to match bins when concatenating
        next_bin = pl.Series([bins[-1] + interval], dtype=bins.dtype)
        bins = pl.concat([bins, next_bin])
    
    # Convert to integer milliseconds and get interval as int
    bins = bins.dt.timestamp('ms')
    interval = int(interval.total_seconds() * 1000)
    return bins, interval

def _filter_times(lf: pl.LazyFrame, unique_times: list[int], ff: bool = False) -> pl.Series:
    # Get time range from data
    time_range = lf.select([
        pl.col('timeMs').min().alias('min_time'),
        pl.col('timeMs').max().cast(pl.Datetime('ms')).alias('max_time')
    ]).collect()
    start_ms = time_range['min_time'][0]
    end_dt = time_range['max_time'][0]
    # Get start of next day (EOD boundary)
    eod_dt = datetime.combine(end_dt.date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
    eod_ms = int(eod_dt.timestamp() * 1000)
    
    # Filter unique_times to data range
    filtered_times = [t for t in unique_times if t >= start_ms and t <= eod_ms]
    
    # For forward fill, add first timestamp after EOD of last day
    if ff and filtered_times:
        # Find first time after EOD
        next_times = [t for t in unique_times if t > eod_ms]
        if next_times:
            filtered_times.append(min(next_times))
    
    # Always return a Series (empty if no times found)
    time_bins = pl.Series('time_bin', sorted(filtered_times) if filtered_times else [], dtype=pl.Int64)
    return time_bins

def bin_ob(lf: pl.LazyFrame, binning: Optional[str] = None, unique_times: Optional[list[int]] = None, count: bool = False, ff: bool = False) -> tuple[pl.LazyFrame, bool]:
    if binning is not None and unique_times is not None:
        raise ValueError("Provide either 'binning' or 'unique_times', not both")
    if binning is None and unique_times is None:
        return (lf, False)
    if binning is not None:
        bins, interval = _make_bins(lf, binning, ff)
    elif unique_times is not None:
        bins = _filter_times(lf, unique_times, ff)
        interval = 60 * 1000 # 1 minute by default
    
    # Handle empty bins case
    if len(bins) == 0:
        return (lf, False)
    
    return (_bin(lf, bins, interval, count, ff), True)

def sink_bins(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Sink time_bin into timeMs by dropping old timeMs and renaming time_bin."""
    return (lf
        .drop('timeMs')
        .rename({'time_bin': 'timeMs'})
    )

# ===============================================================
# Timestamp functions
# ===============================================================

def _get_time_cols(lf: pl.LazyFrame) -> list[str]:
    """Get all time columns from LazyFrame."""
    schema = lf.collect_schema()
    return [
        col for col in schema.names()
        if ('time' in col.lower() or col == 'expiry') and schema[col] in [pl.Int64, pl.Int32, pl.UInt64, pl.UInt32]
    ]

def times_to_dt(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Convert time columns from milliseconds (Int64) to datetime."""
    time_cols = _get_time_cols(lf)
    if not time_cols:
        return lf
    return lf.with_columns([pl.col(col).cast(pl.Datetime('ms')) for col in time_cols])

def times_to_ms(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Convert time columns from datetime to milliseconds (Int64)."""
    time_cols = _get_time_cols(lf)
    if not time_cols:
        return lf
    return lf.with_columns([pl.col(col).dt.timestamp('ms').alias(col) for col in time_cols])

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
    return lf.with_columns(
        pl.col('symbol')
            .str.split('.').list.first() # Remove extension
            .str.split('-')
            .alias('_parts')
    ).with_columns([pl.col('_parts').list.get(3).cast(pl.Int64).alias('strike'), pl.col('_parts').list.get(4).str.to_uppercase().alias('opt_type')]).drop('_parts')

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