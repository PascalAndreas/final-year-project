# Orderbook Data Storage Infrastructure

A lightweight data lake implementation for OKX historical orderbook data with lazy computation and flexible caching.

## Architecture

### Two-Tier Storage

1. **Raw Layer** (`data/okx/orderbook/`)
   - Always stores full depth=5 orderbook snapshots
   - Immutable once written
   - Organized by family/type/date: `BTC-USD/SWAP/2025-09-01.parquet`

2. **Cache Layer** (`data/okx/cache/`)
   - User-named derived transformations
   - Clearable without affecting raw data
   - Organized under cache name: `cache/d1_1m/BTC-USD/SWAP/2025-09-01.parquet`

3. **Manifest** (`data/okx/manifest.sqlite`)
   - SQLite database tracking all files
   - Records: inst_family, inst_type, date, variant (raw or cache name), path, row counts

### File Format

- **Parquet** with zstd compression
- **Directory partitioning** by instrument family, type, and date
- **Column statistics** enabled for efficient queries
- **Atomic writes** using temporary files to prevent corruption

## API

### Setup

```python
from okx.store import OrderbookStore, populate, FEATURES

store = OrderbookStore(
    data_root="data/okx",
    manifest_path="data/okx/manifest.sqlite",
    batch_days=7  # Optional: process dates in batches to avoid memory issues
)
```

The `batch_days` parameter is useful when working with large date ranges (e.g., 2+ months). When set, data is processed in batches of the specified number of days, reducing memory usage.

### Populate Raw Data

```python
from datetime import datetime, date

# Using start/end datetime
populate(
    store,
    inst_family='BTC-USD',
    inst_type='SWAP',
    start=datetime(2025, 9, 1),
    end=datetime(2025, 9, 30),
    max_workers=8,  # Parallel downloads
    verbose=True
)

# Or using explicit date list
populate(
    store,
    inst_family='BTC-USD',
    inst_type='SWAP',
    dates=[date(2025, 9, 1), date(2025, 9, 2), date(2025, 9, 3)],
    max_workers=8
)
```

- Only fetches missing dates (incremental updates)
- Always stores full depth=5 orderbook snapshots
- Downloads dates in parallel for speed (configurable with `max_workers`)
- Provide either `(start, end)` or `dates` parameter

### Query Data

```python
from datetime import datetime

# Cached query
lf = store.get(
    inst_family='BTC-USD',
    inst_type='SWAP',
    start=datetime(2025, 9, 1),
    end=datetime(2025, 9, 5),
    depth=1,           # Trim to depth 1
    binning='1m',      # 1-minute bins
    features=['mid', 'spread'],  # Add computed features
    cache_name='d1_1m_features'  # Cache name (creates if missing)
)
df = lf.collect()  # Polars DataFrame

# Ad-hoc query (no caching)
lf = store.get(
    inst_family='BTC-USD',
    inst_type='SWAP',
    start=datetime(2025, 9, 1),
    end=datetime(2025, 9, 2),
    depth=2,
    binning='5m'
)
```

### Features

**Special features** (built-in with special handling):
- `'tenor'`: Parses instrument symbols to add `expiry` (datetime) and `T` (time to maturity in years) columns. Handles FUTURES, OPTION, and SWAP instruments.
- `'trim'`: Apply depth trimming at this position in the pipeline
- `'bin'`: Apply time binning at this position in the pipeline

**Registry features** (from `FEATURES` dict):
- `mid`: (ask_1_px + bid_1_px) / 2
- `spread`: ask_1_px - bid_1_px
- `rel_spread`: spread / mid
- `imbalance1`: (bid_1_qty - ask_1_qty) / (bid_1_qty + ask_1_qty)
- `imbalance5`: Same as imbalance1 but summing all 5 levels
- `bid_volume`: Sum of bid_1_qty through bid_5_qty
- `ask_volume`: Sum of ask_1_qty through ask_5_qty

**Custom features** (callable functions):
```python
import polars as pl

def add_volatility(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns([
        pl.col('mid').rolling_std(window_size=10).alias('vol_10')
    ])

lf = store.get(
    inst_family='BTC-USD',
    inst_type='SWAP',
    start=datetime(2025, 9, 1),
    end=datetime(2025, 9, 2),
    depth=1,
    binning='1m',
    features=['mid', add_volatility],  # Mix registry + custom
    cache_name='d1_1m_vol'
)
```

**Feature ordering** - Control when transformations are applied:
```python
# Default: features → trim → bin
lf = store.get(..., features=['mid', 'spread'])

# Custom: bin → trim → features (useful for rolling windows over binned data)
lf = store.get(..., features=['bin', 'trim', 'mid', 'spread', add_volatility])
```

Features are flushed (applied) before `'trim'`, `'bin'`, or callable functions to ensure correct execution order.

### Cache Management

```python
# List caches
store.manifest.list_caches()  # ['d1_1m', 'd1_5m_features', ...]

# Clear specific cache
store.clear_cache('d1_1m')

# Clear all caches (keeps raw data)
store.clear_cache()
```

### Derived Data (Recipes)

For complex computations that combine multiple data sources (e.g., forward curves from SWAP + FUTURES), use `get_derived()` with recipe functions:

```python
from datetime import datetime
from functools import partial

def build_forward_curve(store, dates):
    """Recipe signature: (store, dates) -> LazyFrame"""
    if not dates:
        return pl.LazyFrame()
    
    # Fetch SWAP and FUTURES data
    swap = store.get('BTC-USD', 'SWAP', dates=dates, depth=1, 
                     binning='5m', features=['tenor', 'rel_spread'],
                     cache_name='d1_5m_pillars')
    futures = store.get('BTC-USD', 'FUTURES', dates=dates, depth=1,
                        binning='5m', features=['tenor', 'rel_spread'],
                        cache_name='d1_5m_pillars')
    
    # ... complex forward curve computation ...
    return forward_curve_lf

# Query with caching
lf = store.get_derived(
    build_forward_curve,
    start=datetime(2025, 9, 1),
    end=datetime(2025, 9, 5),
    cache_name='forwards_5m'  # Optional: cache the result
)

# Or use functools.partial to configure recipes
from okx.recipes.forwards import build_forwards_pchip

recipe = partial(build_forwards_pchip, binning='5m', lambda_ewma=0.8)
lf = store.get_derived(recipe, start=start, end=end, cache_name='forwards_pchip_5m')
```

**Key features:**
- Recipe signature: `(store, dates) -> pl.LazyFrame`
- Full access to store for fetching multiple sources
- Optional caching with `cache_name`
- Supports both `(start, end)` and `dates` parameters
- Batching handled automatically if `batch_days` is set on store

## Transformation Pipeline

When you call `store.get()` with transformations:

1. **Scan raw** - Polars lazy scan of depth=5 parquet files across requested dates
2. **Apply transformations** (default order: features → trim → bin):
   - **Features** - Apply registry expressions (batched) or custom callables
   - **Trim depth** - Select only bid/ask columns up to specified depth (e.g., depth=1 keeps only bid_1_px, bid_1_qty, etc.)
   - **Bin timestamps** - Use `group_by_dynamic()` on `timeMs` column, take last entry per bin
3. **Time filter** - Filter to exact start/end datetime range
4. **Cache (optional)** - Write transformed data to cache layer (split by date)

All operations are lazy until `.collect()` is called. If caching is enabled, the cache is built incrementally (only missing dates are computed).

## Benefits

1. **Fetch once, query many** - Raw data cached locally, no repeated API calls
2. **Lazy computation** - Only compute what you need when you need it
3. **Flexible caching** - Cache any transformation with a descriptive name
4. **Clean separation** - Raw data immutable, caches clearable
5. **Efficient storage** - Parquet + zstd compression (~10x smaller than CSV)
6. **Fast queries** - Polars streaming + column pruning + predicate pushdown
7. **Parallel downloads** - Fetch multiple dates concurrently for faster population
8. **Incremental caching** - Only compute missing dates when building caches
9. **No lock-in** - Standard Parquet format, readable by any tool

## File Sizes (Typical)

- **Raw depth=5**: ~50-100 MB/day for a single instrument
- **Depth=1, 1-min bins**: ~5-10 MB/day
- **Depth=1, 5-min bins with features**: ~1-2 MB/day

## Implementation Details

### Key Files

- `okx/store.py` - Main storage infrastructure (~480 lines)
  - `Manifest` class - SQLite manifest management
  - `OrderbookStore` class - Main API with batching support
  - `FEATURES` registry - Common feature expressions
  - `populate()` function - Parallel fetching and storage
  - `get()` - Query with transformations and caching
  - `get_derived()` - Execute recipes combining multiple sources

- `okx/helpers.py` - Helper functions
  - `trim_orderbook_polars()` - Depth trimming for Polars
  - `bin_orderbook_polars()` - Time binning for Polars
  - `add_tenor_polars()` - Parse symbols to add expiry and T columns

- `okx/recipes/` - Recipe functions for derived data
  - `forwards.py` - Forward curve construction (PCHIP and Kalman methods)

- `okx/api.py` - Data fetching
  - `fetch_orderbook_lazy()` - Streaming fetch that returns LazyFrame

### Design Decisions

- **Polars-first design** - LazyFrame streaming throughout for memory efficiency
- **Instrument family + type organization** - Flexible grouping (e.g., BTC-USD family includes SWAP, FUTURES, SPOT, OPTIONS)
- **User-named caches** - Explicit cache names for clarity and reproducibility
- **Manual cache invalidation only** - Historical data is immutable
- **Functional API** - `store.get()` with parameters, not builder pattern
- **Lazy by default** - Returns LazyFrame, user calls `.collect()` when ready
- **Atomic writes** - Write to `.parquet.tmp` then rename to avoid corruption
- **Parallel population** - ThreadPoolExecutor for concurrent date downloads
- **Incremental updates** - Manifest tracks what exists, only fetch/compute what's missing
- **Streaming writes** - Use `sink_parquet()` to avoid materializing large datasets
- **Configurable feature ordering** - Allow `'trim'` and `'bin'` in features list for precise control
- **Optional batching** - Process large date ranges in configurable batches to manage memory
- **Recipe system** - `get_derived()` for complex multi-source computations with same caching semantics
- **Flexible date specification** - Accept either `(start, end)` datetime range or explicit `dates` list

## Example Workflow

See `store_example.ipynb` for a complete tutorial.

