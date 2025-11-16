"""
Orderbook data storage with raw depth=5 + derived cache layers.
Organized by inst_family/inst_type combos, not individual instruments.
"""

import sqlite3
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Callable
import polars as pl
from .api import fetch_orderbook_lazy

from tqdm.auto import tqdm
from .features import FEATURES, build_flush_features, sink_bins
from .helpers import _get_function_name

# =============================================================================
# Manifest
# =============================================================================

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


# =============================================================================
# OrderbookStore
# =============================================================================

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

    def _date_range(self, start: Optional[datetime] = None, end: Optional[datetime] = None,
                    dates: Optional[list[date]] = None) -> list[date]:
        """Generate or validate date list: provide either (start, end) or dates; returns list of dates."""
        if dates is not None:
            return dates
        if start is None or end is None:
            raise ValueError("Must provide either 'dates' or both 'start' and 'end'")
        return [start.date() + timedelta(days=i) 
                for i in range((end.date() - start.date()).days)]

    # =============================================================================
    # Storage and Cache Management
    # =============================================================================
    
    def _path_for(self, inst_family: str, inst_type: str, date_obj: date, variant: str = 'raw') -> pathlib.Path:
        """Generate parquet path for family/type/date/variant."""
        if variant == 'raw':
            return (self.root / 'orderbook' / inst_family / inst_type / f'{date_obj.isoformat()}.parquet')
        elif inst_family == '__derived__' and inst_type == '__derived__':
            return (self.root / 'cache' / 'derived' / variant / f'{date_obj.isoformat()}.parquet')
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
        
        # Raw data is sorted by timeMs (enforced at populate time)
        return pl.scan_parquet(paths).set_sorted('timeMs')

    def _write_cache(self, lf: pl.LazyFrame, dates: list[date], cache_name: str,
                     inst_family: str, inst_type: str) -> None:
        """Write LazyFrame to cache (one file per date)."""
        df = lf.collect()
        if df.is_empty():
            return
        
        for date_val in dates:
            date_df = df.filter(pl.from_epoch('timeMs', time_unit='ms').cast(pl.Date) == date_val)
            if date_df.is_empty():
                continue
            
            path = self._path_for(inst_family, inst_type, date_val, cache_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write
            tmp_path = path.with_suffix('.parquet.tmp')
            date_df.write_parquet(str(tmp_path), compression='zstd', statistics=True)
            tmp_path.rename(path)
            
            self.manifest.upsert(ManifestRow(
                inst_family=inst_family, inst_type=inst_type, date=date_val.isoformat(),
                variant=cache_name, path=str(path), rows=len(date_df)
            ))
    
    def _ensure_cached(self, inst_family: str, inst_type: str, dates: list[date],
                       builder: Callable[[list[date]], pl.LazyFrame],
                       cache_name: Optional[str] = None) -> None:
        """Build and cache missing dates."""
        if cache_name is None:
            return
        missing_dates = [d for d in dates 
                        if not self.manifest.have(inst_family, inst_type, d, cache_name)]
        
        if missing_dates:
            lf_missing = builder(missing_dates)
            self._write_cache(lf_missing, missing_dates, cache_name, inst_family, inst_type)
    
    def _load_from_cache(self, inst_family: str, inst_type: str, 
                         dates: list[date], cache_name: str) -> pl.LazyFrame:
        """Load requested dates from cache."""
        paths = [str(self._path_for(inst_family, inst_type, d, cache_name)) 
                 for d in dates if self.manifest.have(inst_family, inst_type, d, cache_name)]
        return pl.scan_parquet(paths) if paths else pl.LazyFrame()

    def clear_cache(self, cache_name: Optional[str] = None):
        """Delete cached files and manifest entries."""
        import shutil
        cache_dir = self.root / 'cache'
        
        if cache_name:
            # Delete specific cache (check both regular and derived locations)
            regular_cache = cache_dir / cache_name
            derived_cache = cache_dir / 'derived' / cache_name
            
            if regular_cache.exists():
                shutil.rmtree(regular_cache)
            if derived_cache.exists():
                shutil.rmtree(derived_cache)
            
            self.manifest.delete_cache(cache_name)
            print(f"Cleared cache: {cache_name}")
        else:
            # Delete all caches
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir()
            
            for cache in self.manifest.list_caches():
                self.manifest.delete_cache(cache)
            print("Cleared all caches")

    def delete_raw(self, inst_family: str, inst_type: str, date_obj: date):
        """Delete raw orderbook file and manifest entry for given family/type/date."""
        path = self._path_for(inst_family, inst_type, date_obj, 'raw')
        if path.exists():
            path.unlink()
        with sqlite3.connect(self.manifest.path) as conn:
            conn.execute("DELETE FROM files WHERE inst_family=? AND inst_type=? AND date=? AND variant='raw'",
                        (inst_family, inst_type, date_obj.isoformat()))

    def migrate(self, transform_fn: Callable[[pl.LazyFrame], pl.LazyFrame], 
                variant: str = 'raw', verbose: bool = True):
        """Apply transformation function to all parquet files in manifest."""
        with sqlite3.connect(self.manifest.path) as conn:
            rows = conn.execute("SELECT inst_family, inst_type, date, path FROM files WHERE variant=?", 
                              (variant,)).fetchall()
        
        for i, (inst_family, inst_type, date_str, path_str) in enumerate(rows):
            if verbose:
                print(f"[{i+1}/{len(rows)}] Migrating {inst_family}/{inst_type}/{date_str}")
            path = pathlib.Path(path_str)
            lf = transform_fn(pl.scan_parquet(path))
            tmp = path.with_suffix('.parquet.tmp')
            lf.sink_parquet(str(tmp), compression='zstd')
            tmp.rename(path)
    
    def inspect(self, fn: Callable[[pl.LazyFrame, str, str, str], any], 
                variant: str = 'raw', verbose: bool = True) -> list[any]:
        """Run arbitrary function over all parquet files in manifest (read-only)."""
        with sqlite3.connect(self.manifest.path) as conn:
            rows = conn.execute("SELECT inst_family, inst_type, date, path FROM files WHERE variant=?", 
                              (variant,)).fetchall()
        
        results = []
        for i, (inst_family, inst_type, date_str, path_str) in enumerate(rows):
            if verbose:
                print(f"[{i+1}/{len(rows)}] Inspecting {inst_family}/{inst_type}/{date_str}")
            path = pathlib.Path(path_str)
            lf = pl.scan_parquet(path)
            result = fn(lf, inst_family, inst_type, date_str)
            results.append(result)
        
        return results
    
    # =============================================================================
    # Data Retrieval
    # =============================================================================

    def _get(self, inst_family: str, inst_type: str, dates: list[date],
             builder: Callable[[list[date]], pl.LazyFrame],
             cache_name: Optional[str] = None,
             batch_days: Optional[int] = None,
             verbose: bool = False, benchmark: bool = False) -> pl.LazyFrame:
        """Helper for get() and get_derived() to handle caching logic with optional batching."""
        # Setup batches (single batch if no batching needed)
        needs_batching = batch_days is not None and len(dates) > batch_days
        batches = [dates[i:i + batch_days] for i in range(0, len(dates), batch_days)] if needs_batching else [dates]
        batch_iter = tqdm(batches, desc="Processing batches", disable=not verbose or not needs_batching)
        
        if benchmark:
            t0 = time.perf_counter()
        if cache_name:
            # Ensure all batches are cached, then load from cache
            for batch_dates in batch_iter:
                self._ensure_cached(inst_family, inst_type, batch_dates, builder, cache_name)
            if benchmark:
                print(f"  [benchmark] Cache building: {time.perf_counter() - t0:.3f}s")
                t0 = time.perf_counter()
            result = self._load_from_cache(inst_family, inst_type, dates, cache_name)
            if benchmark:
                print(f"  [benchmark] Cache loading: {time.perf_counter() - t0:.3f}s")
        else:
            # Build batches directly and concatenate
            batch_results = [builder(batch_dates) for batch_dates in batch_iter]
            result = pl.concat(batch_results) if batch_results else pl.LazyFrame()
            if benchmark:
                print(f"  [benchmark] Direct build: {time.perf_counter() - t0:.3f}s")
        
        return result

    def _apply_transforms(self, lf: pl.LazyFrame, inst_family: str, inst_type: str,
                          depth: Optional[int], binning: Optional[str], 
                          unique_times: Optional[list[int]], 
                          features: list = [], 
                          verbose: bool = False, benchmark: bool = False,
                          benchmark_fn: Optional[Callable[[pl.DataFrame], None]] = None) -> pl.LazyFrame:
        """
        Apply transformations in order specified by features list.
        
        If features contains 'trim' and/or 'bin', those operations are executed
        at that position. Otherwise, default order is: features → trim → bin.
        """
        if lf.collect_schema() == {}:  # Empty LazyFrame
            return lf
        
        # Helper to benchmark individual transformations (for FLUSH_FEATURES and callables)
        def _benchmark_transform(lf: pl.LazyFrame, feat_name: str) -> pl.LazyFrame:
            if not benchmark:
                return lf
            t0 = time.perf_counter()
            df = lf.collect()
            elapsed = time.perf_counter() - t0
            print(f"  [benchmark] {feat_name}: {elapsed:.3f}s")
            # Call user-provided benchmark function if provided
            if benchmark_fn is not None:
                benchmark_fn(df)
            return df.lazy()

        # Helper to flush accumulated feature names with optional benchmarking
        def _flush(lf: pl.LazyFrame, feat_names: list) -> tuple[pl.LazyFrame, list]:
            if feat_names:
                exprs = [FEATURES[name].alias(name) for name in feat_names]
                lf = lf.with_columns(exprs)
                if benchmark:
                    names = ', '.join(feat_names) if len(feat_names) > 1 else feat_names[0]
                    lf = _benchmark_transform(lf, names)
            return lf, []
        
        # Callable features that require flushing before execution
        FLUSH_FEATURES = build_flush_features(inst_type, depth, binning, unique_times)
        
        # Process features list in order
        trim_applied, bin_applied = False, False
        feature_names = []
        
        for feat in features:
            if feat in FLUSH_FEATURES:
                # Flush accumulated features, then apply the transformation
                lf, feature_names = _flush(lf, feature_names)
                lf, was_applied = FLUSH_FEATURES[feat](lf)
                if was_applied:
                    lf = _benchmark_transform(lf, feat)
                # Track trim/bin application for default fallback
                if feat == 'trim':
                    trim_applied = was_applied
                elif feat.startswith('bin'):
                    bin_applied = was_applied
            elif feat in FEATURES:
                # Registry feature - accumulate for parallel execution
                feature_names.append(feat)
            elif callable(feat):
                lf, feature_names = _flush(lf, feature_names)
                lf = feat(lf)
                feat_name = _get_function_name(feat)
                lf = _benchmark_transform(lf, feat_name)
            else:
                if isinstance(feat, str):
                    available = [*FEATURES.keys(), *FLUSH_FEATURES.keys()]
                    raise ValueError(f"Unknown feature '{feat}'. Available: {available}")
                raise ValueError(f"Feature must be string (registry name), callable, got {type(feat)}")
        
        # Final flush
        lf, feature_names = _flush(lf, feature_names)
        
        # Apply trim/bin if not explicitly ordered
        if not trim_applied and depth is not None:
            lf, trim_applied = FLUSH_FEATURES['trim'](lf)
            if trim_applied:
                lf = _benchmark_transform(lf, 'trim (default)')
        if not bin_applied and binning is not None:
            lf, bin_applied = FLUSH_FEATURES['bin'](lf)
            if bin_applied:
                lf = _benchmark_transform(lf, 'bin (default)')
        
        return lf

    def get(self, inst_family: str, inst_type: str, 
            start: Optional[datetime] = None, end: Optional[datetime] = None,
            dates: Optional[list[date]] = None,
            depth: Optional[int] = None, binning: Optional[str] = None, 
            unique_times: Optional[list[int]] = None,
            features: list = [], cache_name: Optional[str] = None,
            batch_days: Optional[int] = None,
            verbose: bool = False, benchmark: bool = False,
            benchmark_fn: Optional[Callable[[pl.DataFrame], None]] = None) -> pl.LazyFrame:
        """
        Get orderbook data with transformations.
        Provide either (start, end) datetimes or a list of dates. If cache_name provided, result is cached.
        Features can include feature strings found in features.py or custom callables.
        """
        if binning is not None and unique_times is not None:
            raise ValueError("Provide either 'binning' or 'unique_times', not both")

        dates = self._date_range(start, end, dates)
        if verbose:
            strategy = f'{binning} binning' if binning is not None else 'provided timestamps'
            print(f"[store] Getting {inst_family}/{inst_type} for {len(dates)} dates (depth={depth}, {strategy}, {len(features)} features)")
        
        # Build batches
        builder = lambda dates_to_build: self._apply_transforms(
            self._scan_raw(inst_family, inst_type, dates_to_build),
            inst_family, inst_type,
            depth, binning, unique_times, features,
            verbose=verbose, benchmark=benchmark, benchmark_fn=benchmark_fn
        )
        result = self._get(inst_family, inst_type, dates, builder, cache_name, batch_days, 
                          verbose, benchmark)
        
        # Handle forward fill boundary conditions
        # Deduplicate if using forward fill with batching (to handle overlapping bins at batch edges)
        using_ff = any('ff' in str(f) for f in features)
        batching = batch_days is not None and len(dates) > batch_days
        if using_ff:
            cols = result.collect_schema().names()
            time_col = 'time_bin' if 'time_bin' in cols else 'timeMs'
            end_dt = datetime.combine(dates[-1], datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
            end_ms = int(end_dt.timestamp() * 1000)
            result = result.filter(pl.col(time_col) <= end_ms)
            if batching:
                result = result.unique(subset=['symbol', time_col], keep='last', maintain_order=True)

        return result
    
    def get_derived(self, recipe: Callable[[object, list[date]], pl.LazyFrame],
                    start: Optional[datetime] = None, end: Optional[datetime] = None,
                    dates: Optional[list[date]] = None,
                    cache_name: Optional[str] = None,
                    batch_days: Optional[int] = None,
                    verbose: bool = False, benchmark: bool = False) -> pl.LazyFrame:
        """
        Get derived data via recipe function with signature (store, dates, verbose) -> LazyFrame.
        Provide either (start, end) datetimes or a list of dates. If cache_name provided, result is cached.
        """
        dates = self._date_range(start, end, dates)
        if verbose:
            print(f"[store] Getting derived data via recipe '{_get_function_name(recipe)}' for {len(dates)} dates")
        builder = lambda dates_to_build: recipe(self, dates_to_build, verbose=verbose)
        return self._get('__derived__', '__derived__', dates, builder, cache_name, batch_days, verbose, benchmark)
    
    # =============================================================================
    # Data Population
    # =============================================================================

    def _log_missing_data(self, date_val: date, inst_family: str, inst_type: str, variant: str):
        """Log missing data to missing.txt file (avoid duplicates)."""
        missing_file = self.root / 'missing.txt'
        entry = f"{date_val.isoformat()},{inst_family},{inst_type},{variant}\n"
        
        # Read existing entries if file exists
        existing_entries = set()
        if missing_file.exists():
            with open(missing_file, 'r') as f:
                existing_entries = set(f.readlines())
        
        # Only append if not already present
        if entry not in existing_entries:
            with open(missing_file, 'a') as f:
                f.write(entry)

    def _fetch_and_store_single_date(self, inst_family: str, inst_type: str, 
                                     date_val: date, verbose: bool = True) -> tuple[date, bool, str]:
        """
        Fetch and store orderbook data for a single date using streaming Polars pipeline.
        
        Returns:
            (date, success, message) tuple
        """
        import shutil
        
        temp_dir = None
        try:
            # Use store's temp directory instead of system temp
            temp_base = self.root / 'temp'
            
            # Fetch as LazyFrame (streaming pipeline, returns temp parquet references)
            lf, temp_dir = fetch_orderbook_lazy(inst_family, inst_type, date_val, temp_base_dir=temp_base)
            
            # Check if empty
            if temp_dir is None or lf.collect_schema() is None or len(lf.collect_schema()) == 0:
                self._log_missing_data(date_val, inst_family, inst_type, 'no data returned from API')
                return (date_val, False, "No data returned from API")
            
            # Sort by timeMs and write to storage using sink (streaming write, no materialization)
            path = self._path_for(inst_family, inst_type, date_val, 'raw')
            path.parent.mkdir(parents=True, exist_ok=True)
            
            tmp_path = path.with_suffix('.parquet.tmp')
            lf.sort('timeMs').sink_parquet(str(tmp_path), compression='zstd')
            tmp_path.rename(path)
            
            # Get row count (lightweight scan)
            row_count = pl.scan_parquet(path).select(pl.len()).collect().item()
            
            if row_count == 0:
                self._log_missing_data(date_val, inst_family, inst_type, 'no data after processing')
                return (date_val, False, "No data after processing")
            
            # Update manifest
            self.manifest.upsert(ManifestRow(
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


    def populate(self, inst_family: str, inst_type: str,
                start: Optional[datetime] = None, end: Optional[datetime] = None,
                dates: Optional[list[date]] = None,
                max_workers: int = 8, verbose: bool = True):
        """
        Populate raw orderbook data for missing dates (always stores depth=5).
        
        Provide either (start, end) or dates. Fetches dates in parallel for faster downloads.
        """
        dates = self._date_range(start, end, dates)
        missing = [d for d in dates if not self.manifest.have(inst_family, inst_type, d, 'raw')]
        
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
                    self._fetch_and_store_single_date,
                    inst_family, inst_type, date_val, verbose
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
