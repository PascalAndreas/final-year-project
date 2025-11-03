"""
Orderbook data storage with raw depth=5 + derived cache layers.
Organized by inst_family/inst_type combos, not individual instruments.
"""

import sqlite3
import pathlib
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Callable
import pandas as pd
import polars as pl
from .api import fetch_market_data, fetch_orderbook_lazy
from .helpers import trim_orderbook_polars, bin_orderbook_polars


# Common feature expressions for derived orderbooks
# 
# NOTE: Registry features are simple Polars expressions that can be batched together.
# They are flushed (applied) before:
#   - 'trim': In case they depend on columns that will be removed
#   - 'bin': In case they depend on tick-by-tick structure
#   - Callables: In case the callable depends on them or transforms the LazyFrame
# 
# If you add a new feature type that requires special handling (e.g., needs flushing
# before other registry features), implement it as a callable instead.
FEATURES = {
    "mid": (pl.col("ask_1_px") + pl.col("bid_1_px")) / 2,
    "spread": pl.col("ask_1_px") - pl.col("bid_1_px"),
    "rel_spread": (pl.col("ask_1_px") - pl.col("bid_1_px")) / ((pl.col("ask_1_px") + pl.col("bid_1_px")) / 2),
    "imbalance1": (pl.col("bid_1_qty") - pl.col("ask_1_qty")) / (pl.col("bid_1_qty") + pl.col("ask_1_qty")),
    "imbalance5": (
        (sum(pl.col(f"bid_{i}_qty") for i in range(1, 6)) - sum(pl.col(f"ask_{i}_qty") for i in range(1, 6)))
        / (sum(pl.col(f"bid_{i}_qty") for i in range(1, 6)) + sum(pl.col(f"ask_{i}_qty") for i in range(1, 6)))
    ),
    "bid_volume": sum(pl.col(f"bid_{i}_qty") for i in range(1, 6)),
    "ask_volume": sum(pl.col(f"ask_{i}_qty") for i in range(1, 6)),
}


@dataclass
class ManifestRow:
    """Metadata for a stored orderbook file."""
    inst_family: str
    inst_type: str
    date: str  # YYYY-MM-DD
    variant: str  # 'raw' or cache name
    path: str
    rows: int


class Manifest:
    """SQLite-backed manifest tracking raw and cached orderbook files."""
    
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS files(
                inst_family TEXT, inst_type TEXT, date TEXT, variant TEXT, path TEXT,
                rows INTEGER,
                PRIMARY KEY(inst_family, inst_type, date, variant)
            );""")
    
    def upsert(self, row: ManifestRow):
        """Insert or update a manifest entry."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("""INSERT INTO files VALUES(?,?,?,?,?,?)
                            ON CONFLICT(inst_family,inst_type,date,variant) DO UPDATE SET
                              path=excluded.path, rows=excluded.rows""",
                         (row.inst_family, row.inst_type, row.date, row.variant, row.path, row.rows))
    
    def have(self, inst_family: str, inst_type: str, date_obj: date, variant: str = 'raw') -> bool:
        """Check if data exists for given family/type/date/variant."""
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute("SELECT 1 FROM files WHERE inst_family=? AND inst_type=? AND date=? AND variant=?",
                               (inst_family, inst_type, date_obj.isoformat(), variant))
            return cur.fetchone() is not None
    
    def list_caches(self) -> list[str]:
        """List all unique cache names (excludes 'raw')."""
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute("SELECT DISTINCT variant FROM files WHERE variant != 'raw'")
            return [row[0] for row in cur.fetchall()]
    
    def delete_cache(self, cache_name: str):
        """Remove all entries for a specific cache."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM files WHERE variant=?", (cache_name,))


class OrderbookStore:
    """Storage manager for raw orderbooks and derived caches."""
    
    def __init__(self, data_root: str | pathlib.Path, manifest_path: str | pathlib.Path):
        """
        Args:
            data_root: Root directory for orderbook/ and cache/ subdirs
            manifest_path: Path to SQLite manifest database
        """
        self.root = pathlib.Path(data_root)
        self.manifest = Manifest(pathlib.Path(manifest_path))
    
    def _path_for(self, inst_family: str, inst_type: str, date_obj: date, variant: str = 'raw') -> pathlib.Path:
        """Generate parquet path for family/type/date/variant."""
        if variant == 'raw':
            return (self.root / 'orderbook' / inst_family / inst_type / f'{date_obj.isoformat()}.parquet')
        else:
            return (self.root / 'cache' / variant / inst_family / inst_type / f'{date_obj.isoformat()}.parquet')
    
    def _scan_raw(self, inst_family: str, inst_type: str, dates: list[date]) -> pl.LazyFrame:
        """Scan raw depth=5 orderbook files for given family/type/dates."""
        paths = []
        for d in dates:
            p = self._path_for(inst_family, inst_type, d, 'raw')
            if p.exists():
                paths.append(str(p))
        
        if not paths:
            return pl.LazyFrame()
        
        return pl.scan_parquet(paths)
    
    def _apply_transforms(self, lf: pl.LazyFrame, depth: Optional[int], 
                          binning: Optional[str], features: Optional[list]) -> pl.LazyFrame:
        """
        Apply transformations in order specified by features list.
        
        If features contains 'trim' and/or 'bin', those operations are executed
        at that position. Otherwise, default order is: features → trim → bin.
        """
        if lf.collect_schema() == {}:  # Empty LazyFrame
            return lf
        
        if not features:
            # No features: apply trim then bin
            if depth is not None:
                lf = trim_orderbook_polars(lf, depth)
            if binning is not None:
                lf = bin_orderbook_polars(lf, binning)
            return lf
        
        # Helper to flush accumulated feature expressions
        def _flush(frame: pl.LazyFrame, exprs: list) -> tuple[pl.LazyFrame, list]:
            if exprs:
                frame = frame.with_columns(exprs)
            return frame, []
        
        # Process features list in order
        trim_applied = False
        bin_applied = False
        feature_exprs = []
        
        for feat in features:
            if feat == 'trim':
                lf, feature_exprs = _flush(lf, feature_exprs)
                if depth is not None:
                    lf = trim_orderbook_polars(lf, depth)
                    trim_applied = True
            elif feat == 'bin':
                lf, feature_exprs = _flush(lf, feature_exprs)
                if binning is not None:
                    lf = bin_orderbook_polars(lf, binning)
                    bin_applied = True
            elif isinstance(feat, str):
                # Registry feature
                if feat not in FEATURES:
                    raise ValueError(f"Unknown feature '{feat}'. Available: {list(FEATURES.keys())}")
                feature_exprs.append(FEATURES[feat].alias(feat))
            elif callable(feat):
                lf, feature_exprs = _flush(lf, feature_exprs)
                lf = feat(lf)
            else:
                raise ValueError(f"Feature must be string (registry name), 'trim', 'bin', or callable, got {type(feat)}")
        
        # Final flush
        lf, feature_exprs = _flush(lf, feature_exprs)
        
        # Apply trim/bin if not explicitly ordered
        if not trim_applied and depth is not None:
            lf = trim_orderbook_polars(lf, depth)
        if not bin_applied and binning is not None:
            lf = bin_orderbook_polars(lf, binning)
        
        return lf
    
    def _write_cache(self, lf: pl.LazyFrame, inst_family: str, inst_type: str, 
                     dates: list[date], cache_name: str):
        """Write transformed LazyFrame to cache (one file per date)."""
        # Collect to split by date
        df = lf.collect()
        
        if df.is_empty():
            return
        
        # Add temporary date column from timeMs (always present)
        df = df.with_columns(
            pl.from_epoch('timeMs', time_unit='ms').cast(pl.Date).alias('_date')
        )
        
        # Write each date separately
        for date_val in dates:
            date_df = df.filter(pl.col('_date') == date_val)
            
            if date_df.is_empty():
                # Skip dates with no data (e.g., OKX missing data days)
                continue
            
            # Remove temporary date column before writing
            date_df = date_df.drop('_date')
            
            path = self._path_for(inst_family, inst_type, date_val, cache_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write
            tmp_path = path.with_suffix('.parquet.tmp')
            date_df.write_parquet(str(tmp_path), compression='zstd', statistics=True)
            tmp_path.rename(path)
            
            # Update manifest
            row_count = len(date_df)
            
            self.manifest.upsert(ManifestRow(
                inst_family=inst_family, inst_type=inst_type, date=date_val.isoformat(), 
                variant=cache_name, path=str(path), rows=row_count
            ))
    
    def get(self, inst_family: str, inst_type: str, start: datetime, end: datetime,
            depth: Optional[int] = None, binning: Optional[str] = None, 
            features: Optional[list] = None, cache_name: Optional[str] = None) -> pl.LazyFrame:
        """
        Get orderbook data with optional transformations.
        
        Args:
            inst_family: e.g., 'BTC-USD'
            inst_type: 'SWAP', 'FUTURES', 'OPTION', 'SPOT'
            start: Start datetime
            end: End datetime
            depth: Trim to depth levels (None = full depth=5)
            binning: Time bin frequency (e.g., '1m', '5m')
            features: List of features/operations in execution order. Can contain:
                     - Feature names from FEATURES registry (e.g., 'mid', 'spread')
                     - Callable functions: func(lf: LazyFrame) -> LazyFrame
                     - 'trim': Apply depth trimming at this position
                     - 'bin': Apply time binning at this position
                     If 'trim'/'bin' not in list, they're applied at the end.
            cache_name: If provided, use/create cache; if None, compute on-the-fly
        
        Returns:
            Polars LazyFrame with requested data
        """
        # Generate date list
        dates = [start.date() + timedelta(days=i) 
                 for i in range((end.date() - start.date()).days + 1)]
        
        # If cache requested, check which dates need building
        if cache_name:
            missing_cache_dates = [d for d in dates 
                                   if not self.manifest.have(inst_family, inst_type, d, cache_name)]
            
            if missing_cache_dates:
                # Build cache only for missing dates
                lf_missing = self._scan_raw(inst_family, inst_type, missing_cache_dates)
                lf_missing = self._apply_transforms(lf_missing, depth, binning, features)
                self._write_cache(lf_missing, inst_family, inst_type, missing_cache_dates, cache_name)
            
            # Load from cache (all requested dates that exist)
            paths = [str(self._path_for(inst_family, inst_type, d, cache_name)) 
                     for d in dates if self.manifest.have(inst_family, inst_type, d, cache_name)]
            lf = pl.scan_parquet(paths) if paths else pl.LazyFrame()
        else:
            # No caching - compute on-the-fly
            lf = self._scan_raw(inst_family, inst_type, dates)
            lf = self._apply_transforms(lf, depth, binning, features)
        
        # Filter by time range
        if lf.collect_schema() != {}:
            # Convert datetime to milliseconds for comparison with timeMs
            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            lf = lf.filter(
                (pl.col('timeMs') >= start_ms) & (pl.col('timeMs') < end_ms)
            )
        
        return lf
    
    def clear_cache(self, cache_name: Optional[str] = None):
        """
        Delete cached files and manifest entries.
        
        Args:
            cache_name: Specific cache to delete, or None to delete all caches
        """
        cache_dir = self.root / 'cache'
        
        if cache_name:
            # Delete specific cache
            target_dir = cache_dir / cache_name
            if target_dir.exists():
                import shutil
                shutil.rmtree(target_dir)
            self.manifest.delete_cache(cache_name)
            print(f"Cleared cache: {cache_name}")
        else:
            # Delete all caches
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                cache_dir.mkdir()
            
            for cache in self.manifest.list_caches():
                self.manifest.delete_cache(cache)
            print("Cleared all caches")


def _fetch_and_store_single_date(
    store: OrderbookStore, inst_family: str, inst_type: str, date_val: date,
    verbose: bool
) -> tuple[date, bool, str]:
    """
    Fetch and store orderbook data for a single date using streaming Polars pipeline.
    
    Returns:
        (date, success, message) tuple
    """
    import shutil
    
    temp_dir = None
    try:
        # Use store's temp directory instead of system temp
        temp_base = store.root / 'temp'
        
        # Fetch as LazyFrame (streaming pipeline, returns temp parquet references)
        lf, temp_dir = fetch_orderbook_lazy(inst_family, inst_type, date_val, temp_base_dir=temp_base)
        
        # Check if empty
        if temp_dir is None or lf.collect_schema() is None or len(lf.collect_schema()) == 0:
            return (date_val, False, "No data returned from API")
        
        # Write to storage using sink (streaming write, no materialization)
        path = store._path_for(inst_family, inst_type, date_val, 'raw')
        path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_path = path.with_suffix('.parquet.tmp')
        lf.sink_parquet(str(tmp_path), compression='zstd')
        tmp_path.rename(path)
        
        # Get row count (lightweight scan)
        row_count = pl.scan_parquet(path).select(pl.len()).collect().item()
        
        if row_count == 0:
            return (date_val, False, "No data after processing")
        
        # Update manifest
        store.manifest.upsert(ManifestRow(
            inst_family=inst_family, inst_type=inst_type, date=date_val.isoformat(),
            variant='raw', path=str(path), rows=row_count
        ))
        
        return (date_val, True, f"{row_count:,} rows")
        
    except Exception as e:
        return (date_val, False, f"Error: {str(e)}")
    
    finally:
        # Clean up temp directory
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def populate(store: OrderbookStore, inst_family: str, inst_type: str, 
             start: datetime, end: datetime, max_workers: int = 8, verbose: bool = True):
    """
    Populate raw orderbook data for missing dates (always stores depth=5).
    Fetches dates in parallel for faster downloads.
    
    Args:
        store: OrderbookStore instance
        inst_family: e.g., 'BTC-USD'
        inst_type: 'SWAP', 'FUTURES', 'OPTION', 'SPOT'
        start: Start datetime (inclusive)
        end: End datetime (exclusive)
        max_workers: Number of concurrent downloads
        verbose: Print progress
    """
    # Determine dates to fetch (end is exclusive)
    num_days = (end.date() - start.date()).days
    dates = [start.date() + timedelta(days=i) for i in range(num_days)]
    missing = [d for d in dates if not store.manifest.have(inst_family, inst_type, d, 'raw')]
    
    if not missing:
        if verbose:
            print(f"✓ {inst_family}/{inst_type}: All dates already cached")
        return
    
    if verbose:
        print(f"Fetching {inst_family}/{inst_type}: {len(missing)} missing dates (from {len(dates)} requested)")
    
    # Fetch dates in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_and_store_single_date,
                store, inst_family, inst_type, date_val, verbose
            ): date_val
            for date_val in missing
        }
        
        for future in tqdm(as_completed(futures), total=len(missing), 
                          desc=f"Downloading {inst_family}/{inst_type}", disable=not verbose):
            date_val, success, message = future.result()
            results.append((date_val, success, message))
            if verbose and not success:
                print(f"  {date_val}: {message}")
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    if verbose:
        print(f"✓ Stored {successful}/{len(missing)} days as raw orderbook data")
        if successful < len(missing):
            failed = [date_val for date_val, success, _ in results if not success]
            print(f"  Failed dates: {failed}")
