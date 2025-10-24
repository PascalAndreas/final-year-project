from datetime import datetime
import pandas as pd
import numpy as np

def bin_orderbook(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Bin orderbook timestamps and keep most recent entry per time bin per symbol.
    Returns filtered df with 'time_bin' and 'timestamp' (binned) columns added.
    freq: pandas frequency string (e.g., '5min', '1min', '30s', '100ms')"""
    # Compute time bins without adding to df (more efficient on large dataframes)
    time_bins = pd.to_datetime(df['timeMs'], unit='ms').dt.floor(freq)
    
    # Group and find most recent entries (only creates temporary groupby object)
    df = df.loc[df.groupby([time_bins, 'symbol'])['timeMs'].idxmax()]
    
    # Now add columns only to filtered df
    df['time_bin'] = pd.to_datetime(df['timeMs'], unit='ms').dt.floor(freq)
    df['timestamp'] = df['time_bin'].astype(np.int64) // 10**6
    
    return df

def parse_option_name(instrument_name: str):
    # Remove file extension if present (e.g., 'BTC-USD-250627-200000-C.OK' -> 'BTC-USD-250627-200000-C')
    instrument = instrument_name.split('.')[0]
    parts = instrument.split('-')
    if len(parts) != 5:
        raise ValueError(f"Invalid instrument name format: {instrument}")
    
    return f"{parts[0]}-{parts[1]}", datetime.strptime(parts[2], '%y%m%d'), int(parts[3]), parts[4].upper()

def parse_future_name(instrument_name: str):
    # Remove file extension if present (e.g., 'BTC-USD-250131.OK' -> 'BTC-USD-250131')
    instrument = instrument_name.split('.')[0]
    parts = instrument.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid instrument name format: {instrument}")
    return f"{parts[0]}-{parts[1]}", datetime.strptime(parts[2], '%y%m%d')

def get_option_combos(df: pd.DataFrame):
    # Extract expiry/strike pairs from valid instruments
    combos_df = pd.DataFrame([
        parse_option_name(inst)[1:3] 
        for inst in df.iloc[:, 0].unique()
    ], columns=['expiry', 'strike'])
    
    return combos_df.drop_duplicates().sort_values(['expiry', 'strike']).reset_index(drop=True)

def standardize_orderbook_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Standardize orderbook column names from FUTURES format to OPTIONS format.
    E.g. askPx1 -> ask_1_px, bidSz2 -> bid_2_sz
    Also adds a symbol column from the filename if not present.
    
    Only standardizes if needed - if columns are already in the correct format,
    returns DataFrame unchanged.
    """
    # Check if columns are already in the correct format
    if all(col.count('_') >= 2 for col in df.columns if col.startswith(('ask', 'bid'))):
        return df
        
    # Add symbol column from filename if not present
    if 'symbol' not in df.columns:
        symbol = filename.split('.csv.gz')[0]
        df['symbol'] = symbol
    
    # Standardize column names
    rename_map = {}
    for col in df.columns:
        if col.startswith(('ask', 'bid')):
            num_str = ''.join(filter(str.isdigit, col))
            if not num_str:
                continue
            
            side = col[:3]
            col_type = col[3:-len(num_str)].lower()
            
            # Convert 'cnt' to 'ordCnt'
            if col_type == 'cnt':
                col_type = 'ordcnt'
                
            rename_map[col] = f"{side}_{num_str}_{col_type}"
            
    return df.rename(columns=rename_map)

def trim_orderbook(df: pd.DataFrame, n_levels: int = 5):
    """
    Trim orderbook DataFrame to keep only top n levels.
    If n_levels=0, calculates simple mid price (bid_1 + ask_1) / 2 and drops all orderbook columns.
    Expects OPTIONS format column names (ask_1_px, bid_1_px, etc.)
    """
    if n_levels == 0:
        # Simple mid price from top bid and ask
        df['mid_price'] = (df['bid_1_px'] + df['ask_1_px']) / 2
        
        # Keep only non-orderbook columns plus mid_price
        cols_to_keep = [
            col for col in df.columns
            if not any(col.startswith(f'{side}_') for side in ['ask', 'bid'])
        ]
        return df[cols_to_keep]
    
    # Standard trimming to n levels
    cols_to_keep = [
        col for col in df.columns
        if not any(col.startswith(f'{side}_') for side in ['ask', 'bid']) or
        (col.split('_')[1].isdigit() and int(col.split('_')[1]) <= n_levels)
    ]
    return df[cols_to_keep]
