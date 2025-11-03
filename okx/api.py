import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .helpers import trim_orderbook, standardize_orderbook_columns
import asyncio
import httpx
import zlib
import polars as pl
import tempfile
from typing import Callable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import threading

BASE_URL = "https://www.okx.com/api/v5/public/market-data-history"

# Global rate limiter for API calls
_api_lock = threading.Lock()
_last_api_call = 0
_min_api_interval = 0.2  # Minimum 200ms between API calls


def _is_rate_limit_error(exception):
    """Check if exception is a 429 rate limit error."""
    return isinstance(exception, ValueError) and "429" in str(exception)


def _rate_limit():
    """Enforce minimum time between API calls."""
    with _api_lock:
        global _last_api_call
        elapsed = time.time() - _last_api_call
        if elapsed < _min_api_interval:
            time.sleep(_min_api_interval - elapsed)
        _last_api_call = time.time()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception(_is_rate_limit_error),
    reraise=True
)
def _get_file_list(module: int | str, inst_type: str, inst_family_list: str,
                   start_ms: int, end_ms: int, date_aggr_type: str = 'daily',
                   include_criterion: Callable = None) -> tuple[list[dict], float]:
    """
    Get file list from OKX API and build download info.
    
    Returns:
        tuple: (download_info_list, total_size_mb)
        - download_info_list: List of dicts with 'url' and 'filename' keys
        - total_size_mb: Sum of file sizes in MB
    """
    # Rate limit API calls
    _rate_limit()
    
    params = {
        'module': str(module),
        'instType': inst_type,
        'dateAggrType': date_aggr_type,
        'begin': str(start_ms),
        'end': str(end_ms)
    }
    
    if inst_type == 'SPOT':
        params['instIdList'] = inst_family_list
    else:
        params['instFamilyList'] = inst_family_list

    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}: {response.text}")
    
    data = response.json()
    if data['code'] != '0':
        raise ValueError(f"API Error: {data['msg']}")

    # Return empty list if no data available
    if not data['data'][0]['details']:
        return [], 0.0
    
    # Build download info (filter by include_criterion if provided)
    download_info = []
    total_size_mb = 0.0
    
    for group in data['data'][0]['details'][0]['groupDetails']:
        # Apply filter if provided
        if include_criterion is not None and not include_criterion(group['filename']):
            continue
        
        download_info.append({
            'url': group['url'],
            'filename': group['filename']
        })
        
        if 'sizeMB' in group:
            total_size_mb += float(group['sizeMB'])
    
    return download_info, total_size_mb

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True
)
async def download_csv_async(url: str, output_path: Path, decompress: bool = True) -> None:
    """
    Download CSV file asynchronously with automatic retry on network errors.
    """
    
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                if decompress:
                    # Use streaming decompressor for gzip
                    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)  # 16 = gzip format
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        decompressed = decompressor.decompress(chunk)
                        if decompressed:
                            f.write(decompressed)
                    # Write any remaining data
                    final = decompressor.flush()
                    if final:
                        f.write(final)
                else:
                    # Download as-is
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        f.write(chunk)


def _get_orderbook_schema() -> dict:
    """Get explicit schema for orderbook CSV parsing (depth=5 only)."""
    schema = {'timeMs': pl.Int64, 'exchTimeMs': pl.Int64}
    
    # Add depth=5 orderbook levels (both OPTIONS and FUTURES formats)
    # Prices: Float64 (precision critical), Quantities: Float64, Counts: Int32
    for i in range(1, 6):
        schema.update({
            f'bid_{i}_px': pl.Float64, f'ask_{i}_px': pl.Float64,
            f'bid_{i}_qty': pl.Float64, f'ask_{i}_qty': pl.Float64,
            f'bid_{i}_ordCnt': pl.Int32, f'ask_{i}_ordCnt': pl.Int32,
            f'bidPx{i}': pl.Float64, f'askPx{i}': pl.Float64,
            f'bidSz{i}': pl.Float64, f'askSz{i}': pl.Float64,
            f'bidCnt{i}': pl.Int32, f'askCnt{i}': pl.Int32,
        })
    
    return schema


def fetch_orderbook_lazy(inst_family: str, inst_type: str, date_val, temp_base_dir: Path = None):
    """
    Fetch orderbook data for a single date using streaming Polars pipeline.
    Returns LazyFrame and temp directory path (caller must cleanup after sink).
    
    Args:
        inst_family: e.g. 'BTC-USD'
        inst_type: 'SWAP', 'FUTURES', 'OPTION', 'SPOT'
        date_val: date object for the day to fetch
        temp_base_dir: Custom base directory for temp files (defaults to system temp)
        
    Returns:
        tuple: (LazyFrame, temp_dir_path) - cleanup temp_dir after using LazyFrame
    """
    # Define include_criterion based on inst_type
    if inst_type == 'SWAP':
        include_criterion = lambda filename: filename.startswith(f'{inst_family}-SWAP')
    elif inst_type == 'FUTURES':
        include_criterion = lambda filename: filename.startswith(f'{inst_family}-')
    else:
        include_criterion = None
    
    # Prepare timestamp range (same start and end works for one day of data)
    dt = datetime.combine(date_val, datetime.min.time()).replace(tzinfo=timezone.utc)
    start_ms = int(dt.timestamp() * 1000)
    end_ms = int(dt.timestamp() * 1000)
    
    # Get file list from API (with filtering applied)
    download_tasks, _ = _get_file_list(
        module=6,
        inst_type=inst_type,
        inst_family_list=inst_family,
        start_ms=start_ms,
        end_ms=end_ms,
        date_aggr_type='daily',
        include_criterion=include_criterion
    )
    
    if not download_tasks:
        return pl.LazyFrame(), None
    
    # Create temp directory for downloads and parts
    if temp_base_dir:
        temp_base_dir = Path(temp_base_dir)
        temp_base_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_base_dir / f'okx_orderbook_{tempfile._get_candidate_names().__next__()}'
        temp_dir.mkdir()
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix='okx_orderbook_'))
    
    # Download all CSVs concurrently
    async def download_all():
        semaphore = asyncio.Semaphore(64)
        
        async def download_one(task):
            async with semaphore:
                csv_filename = task['filename'].replace('.csv.gz', '.csv')
                temp_path = temp_dir / csv_filename
                await download_csv_async(task['url'], temp_path, decompress=True)
                return temp_path, task['filename']
        
        return await asyncio.gather(*[download_one(task) for task in download_tasks])
    
    # Run async downloads (handle existing event loop for Jupyter)
    try:
        loop = asyncio.get_running_loop()
        # Already in event loop (e.g., Jupyter), schedule as task
        import concurrent.futures
        with ThreadPoolExecutor() as executor:
            downloaded_files = executor.submit(
                lambda: asyncio.run(download_all())
            ).result()
    except RuntimeError:
        # No event loop running, use asyncio.run()
        downloaded_files = asyncio.run(download_all())
    
    # Get schema for parsing
    schema = _get_orderbook_schema()
    
    # Define depth=5 columns in standardized format
    depth_5_cols = ['timeMs', 'exchTimeMs', 'symbol'] + [
        f'{side}_{i}_{col}' 
        for i in range(1, 6) 
        for side in ['bid', 'ask'] 
        for col in ['px', 'qty', 'ordCnt']
    ]
    
    # FUTURES rename map: old format -> standardized format
    futures_rename = {
        f'{side}{old}{i}': f'{side.lower()}_{i}_{new}'
        for i in range(1, 6)
        for side in ['bid', 'ask']
        for old, new in [('Px', 'px'), ('Sz', 'qty'), ('Cnt', 'ordCnt')]
    }
    
    # Stream each CSV to parquet part
    part_files = []
    for csv_path, filename in downloaded_files:
        # Extract symbol from filename
        symbol = filename.split('.csv.gz')[0]
        
        # Lazy scan CSV
        lf = pl.scan_csv(
            csv_path,
            schema_overrides=schema,
            ignore_errors=True  # Handle OKX's malformed values (e.g. "115334.9.5")
        )
        
        # Standardize FUTURES columns (lazy) - only rename columns that exist
        if inst_type == 'FUTURES':
            # Get actual columns in the CSV
            actual_cols = lf.collect_schema().names()
            # Only apply renames for columns that exist (handles both old and new formats)
            applicable_renames = {old: new for old, new in futures_rename.items() if old in actual_cols}
            if applicable_renames:
                lf = lf.rename(applicable_renames)
        
        # Always add symbol from filename (more reliable than reading from CSV)
        lf = lf.with_columns(pl.lit(symbol).alias('symbol'))
        
        # Ensure all depth 5 columns exist (fill missing with nulls for schema consistency)
        # This is critical for OPTIONS where different instruments may have different depths
        actual_cols = set(lf.collect_schema().names())
        missing_cols = [c for c in depth_5_cols if c not in actual_cols]
        
        if missing_cols:
            # Add missing columns as nulls with appropriate types
            null_exprs = []
            for col in missing_cols:
                if 'Ms' in col:
                    null_exprs.append(pl.lit(None, dtype=pl.Int64).alias(col))
                elif 'ordCnt' in col or 'Cnt' in col:
                    null_exprs.append(pl.lit(None, dtype=pl.Int32).alias(col))
                elif col == 'symbol':
                    null_exprs.append(pl.lit(None, dtype=pl.String).alias(col))
                else:  # price/quantity columns
                    null_exprs.append(pl.lit(None, dtype=pl.Float64).alias(col))
            lf = lf.with_columns(null_exprs)
        
        # Select depth 5 columns in consistent order
        lf = lf.select(depth_5_cols)
        
        # Stream to parquet part (never materialize full CSV in memory)
        part_path = temp_dir / f'part_{len(part_files)}.parquet'
        lf.sink_parquet(part_path, compression='zstd')
        part_files.append(part_path)
        
    # Return lazy scan of all parts (memory-efficient concatenation)
    if not part_files:
        return pl.LazyFrame(), None
    
    return pl.scan_parquet(part_files), temp_dir

def _create_date_chunks(
    start_date: datetime,
    end_date: datetime,
    module: int | str,
    date_aggr_type: str
) -> list[tuple[int, int]]:
    # Determine timezone based on module
    tz_offset = timezone.utc if int(module) == 6 else timezone(timedelta(hours=8))
    
    # Ensure datetime objects are timezone-aware in the API's timezone
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=tz_offset)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=tz_offset)
    
    # Set max days based on module and aggregation type
    max_days = 1 if int(module) == 6 else (20 if date_aggr_type == 'daily' else 500)
    
    # Split into date ranges (API uses inclusive dates, only date portion matters)
    ranges = []
    current = start_date
    while current < end_date:
        # Calculate chunk end: add (max_days - 1) to get max_days inclusive
        chunk_end = current + timedelta(days=max_days - 1, hours=23, minutes=59, seconds=59)
        chunk_end = min(chunk_end, end_date)
        ranges.append((int(current.timestamp() * 1000), int(chunk_end.timestamp() * 1000)))
        # Move to start of next day after chunk_end
        current = chunk_end + timedelta(seconds=1)
        current = current.replace(hour=0, minute=0, second=0, microsecond=0)
    
    return ranges


def fetch_market_data(
    module: int | str,
    inst_type: str,
    inst_family_list: str,
    start_date: datetime,
    end_date: datetime,
    date_aggr_type: str = 'daily',
    delay: float = 0.2,
    depth: int = 5,
    verbose: bool = True,
    include_criterion: callable = None,
    process_fn: callable = None,
    max_workers: int = 32
) -> pd.DataFrame:
    # Create date chunks based on API timezone and limits
    ranges = _create_date_chunks(start_date, end_date, module, date_aggr_type)
    
    if verbose:
        print(f"Fetching {inst_type} data (module={module}) for {inst_family_list}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Split into {len(ranges)} requests")
    
    # Fetch data for each range
    all_dfs = []
    total_downloads = 0
    total_size_mb = 0.0
    
    for i, (begin_ms, end_ms) in enumerate(tqdm(ranges, desc="Fetching data", disable=not verbose)):
        # Get file list from API (with filtering applied)
        try:
            download_info, size_mb = _get_file_list(
                module=module,
                inst_type=inst_type,
                inst_family_list=inst_family_list,
                start_ms=begin_ms,
                end_ms=end_ms,
                date_aggr_type=date_aggr_type,
                include_criterion=include_criterion
            )
        except ValueError as e:
            print(f"Error fetching file list: {e}")
            continue
        
        if not download_info:
            if verbose:
                print(f"Fetch #{i+1}/{len(ranges)}: No files available for this date range")
            continue
        
        total_downloads += len(download_info)
        total_size_mb += size_mb
            
        if verbose:
            print(f"Fetch #{i+1}/{len(ranges)}: {len(download_info)} files found | Total: {total_downloads} files, {total_size_mb:.2f} MB")
        
        # Download CSVs with progress bar
        def download_csv(info, module):
            try:
                if int(module) == 6: # Module 6 uses chunked reading with native pandas gzip handling
                    chunk_iterator = pd.read_csv(
                        info['url'], 
                        compression='gzip',
                        chunksize=100000
                    )
                    
                    trimmed_chunks = []
                    for chunk in chunk_iterator:
                        # Standardize columns if needed (must be done before trim)
                        if inst_type == 'FUTURES':
                            chunk = standardize_orderbook_columns(chunk, filename=info['filename'])
                        
                        # Trim orderbook immediately to reduce memory
                        chunk = trim_orderbook(chunk, depth)
                        trimmed_chunks.append(chunk)
                    
                    # Combine trimmed chunks
                    df = pd.concat(trimmed_chunks, ignore_index=True)
                else: # Module != 6 doesn't use gzip, process normally
                    df = pd.read_csv(info['url'])
                
                if process_fn is not None: # Apply post-processing function if provided
                    df = process_fn(df)

                return df
            except Exception as e:
                print(f"Error downloading CSV: {e}")
                print(f"URL: {info['url']}")
                print(f"Filename: {info['filename']}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_csv, info, module) for info in download_info]
            for future in tqdm(as_completed(futures), total=len(download_info), desc="Downloading CSVs", disable=not verbose, leave=False):
                df = future.result()
                if df is not None:
                    all_dfs.append(df)
        
        if i < len(ranges) - 1:
            time.sleep(delay)
    
    if not all_dfs:
        if verbose:
            print("No data found")
        return pd.DataFrame()
    
    # Combine all dataframes, sort by created_time and reset index
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    if verbose:
        print(f"✓ Successfully fetched {len(combined_df)} records")
    
    return combined_df