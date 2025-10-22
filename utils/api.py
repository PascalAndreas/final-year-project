import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.helpers import trim_orderbook

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
    verbose: bool = True
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
        if 'data' in data and data['data']:
            # Collect all CSV URLs and sizes
            urls = []
            for data_item in data['data']:
                if 'details' not in data_item:
                    continue
                for detail in data_item['details']:
                    if 'groupDetails' not in detail:
                        continue
                    for group in detail['groupDetails']:
                        if 'url' in group:
                            urls.append(group['url'])
                            # Track size if available
                            if 'sizeMB' in group:
                                total_size_mb += float(group['sizeMB'])
            
            total_downloads += len(urls)
            
            if verbose:
                print(f"Fetch #{i+1}/{len(ranges)}: {len(urls)} files found | Total: {total_downloads} files, {total_size_mb:.2f} MB")
            
            # Download CSVs with progress bar
            def download_csv(url, module):
                try:
                    if int(module) == 6:
                        return trim_orderbook(pd.read_csv(url, compression='gzip'), depth)
                    else:
                        return pd.read_csv(url)
                except Exception as e:
                    print(f"Error downloading CSV: {e}")
                    print(f"URL: {url}")
                    return None

            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(download_csv, url, module) for url in urls]
                for future in tqdm(as_completed(futures), total=len(urls), desc="Downloading CSVs", disable=not verbose, leave=False):
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