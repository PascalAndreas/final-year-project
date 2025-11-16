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
        'bin': (lambda lf: (bin_ob(lf, binning), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_count': (lambda lf: (bin_ob(lf, binning, count=True), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_ff': (lambda lf: (bin_ob(lf, binning, ff=True), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_count_ff': (lambda lf: (bin_ob(lf, binning, count=True, ff=True), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_ff_count': (lambda lf: (bin_ob(lf, binning, count=True, ff=True), True)) if binning is not None else (lambda lf: (lf, False)),
        
        # V2 binning (uses join_asof) - infers time bins from data
        'bin_v2': (lambda lf: (bin_ob_v2(lf, binning), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_v2_count': (lambda lf: (bin_ob_v2(lf, binning, count=True), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_v2_ff': (lambda lf: (bin_ob_v2(lf, binning, ff=True), True)) if binning is not None else (lambda lf: (lf, False)),
        'bin_v2_ff_count': (lambda lf: (bin_ob_v2(lf, binning, ff=True, count=True), True)) if binning is not None else (lambda lf: (lf, False)),
        
        # Bin to explicit times - infers range from data
        'bin_to_times': (lambda lf: (bin_by_times(lf, unique_times), True)) if unique_times is not None else (lambda lf: (lf, False)),
        'bin_to_times_count': (lambda lf: (bin_by_times(lf, unique_times, count=True), True)) if unique_times is not None else (lambda lf: (lf, False)),
        'bin_to_times_ff': (lambda lf: (bin_by_times(lf, unique_times, ff=True), True)) if unique_times is not None else (lambda lf: (lf, False)),
        'bin_to_times_ff_count': (lambda lf: (bin_by_times(lf, unique_times, ff=True, count=True), True)) if unique_times is not None else (lambda lf: (lf, False)),
        
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
        .with_columns(pl.col('_ts').dt.timestamp('ms').alias('time_bin'))
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

def generate_time_bins(start_ms: int, end_ms: int, freq: str, include_next: bool = False) -> pl.Series:
    """
    Generate time bins from start to end at given frequency.
    
    Args:
        start_ms: Start timestamp (ms)
        end_ms: End timestamp (ms)
        freq: Polars duration string (e.g., '5m', '1m', '30s')
        include_next: If True, include one additional bin after end_ms
    
    Returns:
        Series of Int64 timestamps (ms)
    """
    # Create datetime range
    start_dt = pl.from_epoch(start_ms, time_unit='ms')
    end_dt = pl.from_epoch(end_ms, time_unit='ms')
    
    # Generate bins using date_range
    bins = pl.datetime_range(
        start_dt,
        end_dt,
        interval=freq,
        closed='right',
        eager=True
    )
    
    # Add one more bin if requested
    if include_next:
        next_bin = bins[-1] + pl.duration(microseconds=pl.duration(freq).dt.total_microseconds()[0])
        bins = pl.concat([bins, pl.Series([next_bin])])
    
    # Convert to milliseconds
    return bins.dt.timestamp('ms')

def bin_ob_v2(lf: pl.LazyFrame, freq: str, count: bool = False, ff: bool = False) -> pl.LazyFrame:
    """
    Bin orderbook timestamps using join_asof (v2 implementation).
    
    This version is more efficient than bin_ob() as it avoids datetime casting.
    Time bins are inferred from the data range automatically.
    
    Args:
        freq: Polars duration string (e.g., '5m', '1m', '30s')
        count: Add 'bin_count' column with number of entries in each bin
        ff: Forward fill to create complete (symbol, time_bin) grid
    
    Returns:
        LazyFrame with 'time_bin' column added (and 'bin_count' if count=True)
    """
    # Get time range from data and generate bins
    time_range = lf.select([
        pl.col('timeMs').min().alias('min_time'),
        pl.col('timeMs').max().alias('max_time')
    ]).collect()
    start_ms = time_range['min_time'][0]
    end_ms = time_range['max_time'][0]
    
    # For forward fill, extend to first bin of next day to handle batch edges
    include_next = ff
    time_bins = generate_time_bins(start_ms, end_ms, freq, include_next=include_next)
    time_bins_lf = pl.DataFrame({'time_bin': time_bins}).lazy()
    
    # Sort input data
    lf = lf.sort(['symbol', 'timeMs'])
    
    if ff:
        # Create complete grid: all symbols × all time bins
        symbols = lf.select('symbol').unique()
        complete_grid = (
            symbols
            .join(time_bins_lf, how='cross')
            .sort(['symbol', 'time_bin'])
        )
        
        # Join data to grid using asof, then forward fill within each symbol
        binned = (
            complete_grid
            .join_asof(
                lf.with_columns(pl.col('timeMs').alias('time_bin')),
                on='time_bin',
                by='symbol',
                strategy='backward'
            )
            .with_columns(
                pl.all().exclude(['symbol', 'time_bin']).forward_fill().over('symbol')
            )
        )
    else:
        # Create symbol × time_bin grid for join
        symbols = lf.select('symbol').unique()
        grid = symbols.join(time_bins_lf, how='cross').sort(['symbol', 'time_bin'])
        
        # Join using asof to get last value before each bin
        binned = (
            grid
            .join_asof(
                lf.with_columns(pl.col('timeMs').alias('time_bin')),
                on='time_bin',
                by='symbol',
                strategy='backward'
            )
        )
    
    # Add bin_count if requested
    if count:
        # Count how many original rows fall into each bin
        bin_counts = (
            lf
            .join_asof(
                time_bins_lf,
                left_on='timeMs',
                right_on='time_bin',
                strategy='backward'
            )
            .group_by(['symbol', 'time_bin'])
            .agg(pl.len().alias('bin_count'))
        )
        binned = binned.join(bin_counts, on=['symbol', 'time_bin'], how='left')
        binned = binned.with_columns(pl.col('bin_count').fill_null(0))
    
    return binned

def bin_by_times(lf: pl.LazyFrame, unique_times: list[int], ff: bool = False, count: bool = False) -> pl.LazyFrame:
    """
    Bin orderbook to specific timestamps.
    
    Automatically filters unique_times to the data range. For forward fill, extends
    to the first timestamp after EOD of the last day in the data.
    
    Args:
        unique_times: List of Int64 timestamps (ms) to bin to
        ff: Forward fill to create complete grid
        count: Add 'bin_count' column
    
    Returns:
        LazyFrame with 'time_bin' column
    """
    # Get time range from data
    time_range = lf.select([
        pl.col('timeMs').min().alias('min_time'),
        pl.col('timeMs').max().alias('max_time')
    ]).collect()
    start_ms = time_range['min_time'][0]
    end_ms = time_range['max_time'][0]
    
    # Filter unique_times to data range
    filtered_times = [t for t in unique_times if t >= start_ms and t <= end_ms]
    
    # For forward fill, add first timestamp after EOD of last day
    if ff and filtered_times:
        # Get EOD of max day (23:59:59.999)
        from datetime import datetime, timezone
        max_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
        eod_ms = int(datetime(max_dt.year, max_dt.month, max_dt.day, 23, 59, 59, 999000, 
                              tzinfo=timezone.utc).timestamp() * 1000)
        
        # Find first time after EOD
        next_times = [t for t in unique_times if t > eod_ms]
        if next_times:
            filtered_times.append(min(next_times))
    
    if not filtered_times:
        return lf.with_columns(pl.lit(None, dtype=pl.Int64).alias('time_bin'))
    
    time_bins = pl.Series('time_bin', sorted(filtered_times), dtype=pl.Int64)
    time_bins_lf = pl.DataFrame({'time_bin': time_bins}).lazy()
    
    # Sort input data
    lf = lf.sort(['symbol', 'timeMs'])
    
    if ff:
        # Create complete grid: all symbols × all time bins
        symbols = lf.select('symbol').unique()
        complete_grid = (
            symbols
            .join(time_bins_lf, how='cross')
            .sort(['symbol', 'time_bin'])
        )
        
        # Join data to grid using asof, then forward fill within each symbol
        binned = (
            complete_grid
            .join_asof(
                lf.with_columns(pl.col('timeMs').alias('time_bin')),
                on='time_bin',
                by='symbol',
                strategy='backward'
            )
            .with_columns(
                pl.all().exclude(['symbol', 'time_bin']).forward_fill().over('symbol')
            )
        )
    else:
        # Create symbol × time_bin grid for join
        symbols = lf.select('symbol').unique()
        grid = symbols.join(time_bins_lf, how='cross').sort(['symbol', 'time_bin'])
        
        # Join using asof to get last value before each bin
        binned = (
            grid
            .join_asof(
                lf.with_columns(pl.col('timeMs').alias('time_bin')),
                on='time_bin',
                by='symbol',
                strategy='backward'
            )
        )
    
    # Add bin_count if requested
    if count:
        bin_counts = (
            lf
            .join_asof(
                time_bins_lf,
                left_on='timeMs',
                right_on='time_bin',
                strategy='backward'
            )
            .group_by(['symbol', 'time_bin'])
            .agg(pl.len().alias('bin_count'))
        )
        binned = binned.join(bin_counts, on=['symbol', 'time_bin'], how='left')
        binned = binned.with_columns(pl.col('bin_count').fill_null(0))
    
    return binned

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