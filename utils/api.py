import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.helpers import trim_orderbook, standardize_orderbook_columns

BASE_URL = "https://www.okx.com/api/v5/public/market-data-history"

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
        params = {
            'module': str(module),
            'instType': inst_type,
            'instFamilyList': inst_family_list,
            'dateAggrType': date_aggr_type,
            'begin': str(begin_ms),
            'end': str(end_ms)
        }
        
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code} - {response.text}")
            continue
        
        data = response.json()
        
        if data['code'] != '0':
            print(f"API Error: {data['msg']}")
            continue
        
        # Parse response and download CSVs
        download_info = []
        
        # Process group details
        for group in data['data'][0]['details'][0]['groupDetails']:
            # Apply include_criterion filter if provided
            if include_criterion is not None:
                if not include_criterion(group['filename']):
                    continue
            # Default behavior for futures - filter by filename prefix
            elif inst_type == 'FUTURES':
                # Only include files that start with the specified instrument family
                if not group['filename'].startswith(f"{inst_family_list}-"):
                    continue
            
            download_info.append({
                'url': group['url'],
                'filename': group['filename']
            })
            if 'sizeMB' in group:
                total_size_mb += float(group['sizeMB'])
        
        total_downloads += len(download_info)
            
        if verbose:
            print(f"Fetch #{i+1}/{len(ranges)}: {len(download_info)} files found | Total: {total_downloads} files, {total_size_mb:.2f} MB")
        
        # Download CSVs with progress bar
        def download_csv(info, module):
            try:
                if int(module) == 6:
                    if inst_type == 'FUTURES':
                        df = standardize_orderbook_columns(pd.read_csv(info['url'], compression='gzip'), filename=info['filename'])
                    else:
                        df = pd.read_csv(info['url'], compression='gzip')
                    df = trim_orderbook(df, depth)
                else:
                    df = pd.read_csv(info['url'])
                
                # Apply post-processing function if provided
                if process_fn is not None:
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