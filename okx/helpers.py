from datetime import datetime, timezone
from typing import Callable
from functools import partial
import pandas as pd
import numpy as np
import polars as pl

# ===============================================================
# Helper functions
# ===============================================================

def _get_function_name(func: Callable) -> str:
    """Extract function name from function, handles nested functools.partial."""
    while isinstance(func, partial):
        func = func.func
    func_name = func.__name__
    return func_name

# ===============================================================
# Name parsing functions - superceded by polars features, slated for deprecation
# ===============================================================

def parse_option_name(instrument_name: str):
    """Parse option instrument name into components."""
    # Remove file extension if present (e.g., 'BTC-USD-250627-200000-C.OK' -> 'BTC-USD-250627-200000-C')
    instrument = instrument_name.split('.')[0]
    parts = instrument.split('-')
    if len(parts) != 5:
        raise ValueError(f"Invalid instrument name format: {instrument}")
    
    # Parse date and set expiry time to 08:00 UTC
    expiry_datetime = datetime.strptime(parts[2], '%y%m%d').replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    
    return f"{parts[0]}-{parts[1]}", expiry_datetime, int(parts[3]), parts[4].upper()

def parse_future_name(instrument_name: str):
    """Parse futures instrument name into components."""
    # Remove file extension if present (e.g., 'BTC-USD-250131.OK' -> 'BTC-USD-250131')
    instrument = instrument_name.split('.')[0]
    parts = instrument.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid instrument name format: {instrument}")
    
    # Parse date and set expiry time to 08:00 UTC
    expiry_datetime = datetime.strptime(parts[2], '%y%m%d').replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    
    return f"{parts[0]}-{parts[1]}", expiry_datetime

# ===============================================================
# LEGACY PANDAS FUNCTIONS - slated for deprecation
# ===============================================================

def get_option_combos(df: pl.DataFrame | pl.LazyFrame):
    """Extract unique expiry/strike combinations from options data.
    
    Args:
        df: Polars DataFrame/LazyFrame with 'symbol' column containing option instruments
        
    Returns:
        Polars DataFrame with 'expiry' and 'strike' columns
    """
    # Collect if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    # Get unique symbols
    symbols = df.select('symbol').unique().to_series().to_list()
    
    # Parse each symbol
    combos = []
    for symbol in symbols:
        try:
            _, expiry, strike, _ = parse_option_name(symbol)
            combos.append({'expiry': expiry, 'strike': strike})
        except ValueError:
            continue  # Skip invalid symbols
    
    # Create and sort DataFrame
    return (pl.DataFrame(combos)
            .unique()
            .sort(['expiry', 'strike']))

def trim_orderbook(df: pd.DataFrame, n_levels: int = 5):
    """
    Trim orderbook DataFrame to keep only top n levels.
    If n_levels=0, calculates simple mid price (bid_1 + ask_1) / 2 and drops all orderbook columns.
    Expects OPTIONS format column names (ask_1_px, bid_1_px, etc.)
    """
    if n_levels == 0:
        # Return empty if required columns don't exist
        if 'bid_1_px' not in df.columns or 'ask_1_px' not in df.columns:
            base_cols = [col for col in df.columns if not any(col.startswith(f'{side}_') for side in ['ask', 'bid'])]
            return pd.DataFrame(columns=base_cols + ['mid_price'])
        
        # Filter out rows with NaN bid or ask
        df = df[df['bid_1_px'].notna() & df['ask_1_px'].notna()].copy()
        
        # Return empty if no valid rows
        if df.empty:
            base_cols = [col for col in df.columns if not any(col.startswith(f'{side}_') for side in ['ask', 'bid'])]
            return pd.DataFrame(columns=base_cols + ['mid_price'])
        
        # Calculate mid price and keep only non-orderbook columns
        df['mid_price'] = (df['bid_1_px'] + df['ask_1_px']) / 2
        cols_to_keep = [col for col in df.columns if not any(col.startswith(f'{side}_') for side in ['ask', 'bid'])]
        return df[cols_to_keep]
    
    # Standard trimming to n levels
    cols_to_keep = [
        col for col in df.columns
        if not any(col.startswith(f'{side}_') for side in ['ask', 'bid']) or
        (col.split('_')[1].isdigit() and int(col.split('_')[1]) <= n_levels)
    ]
    return df[cols_to_keep]

def bin_orderbook(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Bin orderbook timestamps and keep most recent entry per time bin per symbol.
    Returns filtered df with 'time_bin' and 'timestamp' (binned) columns added.
    freq: pandas frequency string (e.g., '5min', '1min', '30s', '100ms')"""
    # Compute time bins without adding to df (more efficient on large dataframes)
    time_bins = pd.to_datetime(df['timeMs'], unit='ms').dt.ceil(freq)
    
    # Group and find most recent entries (only creates temporary groupby object)
    df = df.loc[df.groupby([time_bins, 'symbol'])['timeMs'].idxmax()]
    
    # Now add columns only to filtered df
    df['time_bin'] = pd.to_datetime(df['timeMs'], unit='ms').dt.ceil(freq)
    df['timestamp'] = df['time_bin'].astype(np.int64) // 10**6
    
    return df

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



