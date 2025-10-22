import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from tqdm.auto import tqdm

# Base URL for historical data
BASE_URL = "https://history.deribit.com/api/v2/public"

def get_trades(currency, type, start_timestamp, end_timestamp=None):
    url = f"{BASE_URL}/get_last_trades_by_currency"
    params = {
        "currency": currency,
        "kind": type,
        "start_timestamp": start_timestamp,
        **({"end_timestamp": end_timestamp} if end_timestamp else {}),
        "sorting": "asc",
        "count": 10000  
    }
    response = requests.get(url, params=params)
    fetch_count = 1
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    
    df = pd.DataFrame(response.json()['result']['trades'])
    
    # Use current time if no end_timestamp provided
    effective_end_timestamp = end_timestamp if end_timestamp else int(datetime.now().timestamp() * 1000)
    time_range = effective_end_timestamp - start_timestamp
    
    # Progress bar based on timestamp
    with tqdm(total=time_range, desc=f"Fetching {currency} {type} trades", unit_scale=True) as pbar:
        pbar.update(0)
        
        while response.json()['result']['has_more']:
            current_timestamp = response.json()['result']['trades'][-1]['timestamp']
            progress = current_timestamp - start_timestamp
            pbar.update(progress - pbar.n)
            
            params['start_timestamp'] = current_timestamp
            response = requests.get(url, params=params)
            fetch_count += 1
            df = pd.concat([df, pd.DataFrame(response.json()['result']['trades'])])
        
        # Complete the progress bar
        pbar.update(time_range - pbar.n)
    
    # Drop duplicates and report statistics
    original_count = len(df)
    df = df.drop_duplicates(subset=['trade_id'])
    duplicates_dropped = original_count - len(df)
    
    print(f"✓ Completed {fetch_count} API fetches")
    print(f"✓ Dropped {duplicates_dropped} duplicate trade(s)")
    
    return df

def get_trades_by_instrument(instrument_name, start_timestamp, end_timestamp):
    url = f"{BASE_URL}/get_last_trades_by_instrument_and_time"
    params = {
        "instrument_name": instrument_name,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "sorting": "asc",
        "count": 10000
    }
    response = requests.get(url, params=params)
    fetch_count = 1
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    
    df = pd.DataFrame(response.json()['result']['trades'])
    
    # Use current time if no end_timestamp provided
    effective_end_timestamp = end_timestamp if end_timestamp else int(datetime.now().timestamp() * 1000)
    time_range = effective_end_timestamp - start_timestamp
    
    # Progress bar based on timestamp
    with tqdm(total=time_range, desc=f"Fetching {instrument_name} trades", unit_scale=True) as pbar:
        pbar.update(0)
        
        while response.json()['result']['has_more']:
            current_timestamp = response.json()['result']['trades'][-1]['timestamp']
            progress = current_timestamp - start_timestamp
            pbar.update(progress - pbar.n)
            
            params['start_timestamp'] = current_timestamp
            response = requests.get(url, params=params)
            fetch_count += 1
            df = pd.concat([df, pd.DataFrame(response.json()['result']['trades'])])
        
        # Complete the progress bar
        pbar.update(time_range - pbar.n)
    
    # Drop duplicates and report statistics
    original_count = len(df)
    df = df.drop_duplicates(subset=['trade_id'])
    duplicates_dropped = original_count - len(df)
    
    print(f"✓ Completed {fetch_count} API fetches")
    print(f"✓ Dropped {duplicates_dropped} duplicate trade(s)")
    
    return df