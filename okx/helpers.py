from datetime import datetime
import pandas as pd
import numpy as np
import polars as pl

def parse_option_name(instrument_name: str):
    """Parse option instrument name into components."""
    # Remove file extension if present (e.g., 'BTC-USD-250627-200000-C.OK' -> 'BTC-USD-250627-200000-C')
    instrument = instrument_name.split('.')[0]
    parts = instrument.split('-')
    if len(parts) != 5:
        raise ValueError(f"Invalid instrument name format: {instrument}")
    
    # Parse date and set expiry time to 08:00 UTC (timezone-naive for pandas compatibility)
    expiry_datetime = datetime.strptime(parts[2], '%y%m%d').replace(hour=8, minute=0, second=0, microsecond=0)
    
    return f"{parts[0]}-{parts[1]}", expiry_datetime, int(parts[3]), parts[4].upper()

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

def standardize_orderbook_columns_polars(df: pl.DataFrame, filename: str) -> pl.DataFrame:
    """
    Standardize orderbook column names from FUTURES format to OPTIONS format (Polars version).
    E.g. askPx1 -> ask_1_px, bidSz2 -> bid_2_sz
    Also adds a symbol column from the filename if not present.
    
    Only standardizes if needed - if columns are already in the correct format,
    returns DataFrame unchanged.
    """
    # Check if columns are already in the correct format
    orderbook_cols = [col for col in df.columns if col.startswith(('ask', 'bid'))]
    if orderbook_cols and all(col.count('_') >= 2 for col in orderbook_cols):
        # Still need to check for symbol column
        if 'symbol' not in df.columns:
            symbol = filename.split('.csv.gz')[0]
            df = df.with_columns(pl.lit(symbol).alias('symbol'))
        return df
    
    # Add symbol column from filename if not present
    if 'symbol' not in df.columns:
        symbol = filename.split('.csv.gz')[0]
        df = df.with_columns(pl.lit(symbol).alias('symbol'))
    
    # Build rename mapping
    rename_map = {}
    for col in df.columns:
        if col.startswith(('ask', 'bid')):
            num_str = ''.join(filter(str.isdigit, col))
            if not num_str:
                continue
            
            side = col[:3]
            col_type = col[3:-len(num_str)].lower()
            
            if col_type == 'cnt':
                col_type = 'ordcnt'
            
            rename_map[col] = f"{side}_{num_str}_{col_type}"
    
    return df.rename(rename_map)

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

def trim_orderbook_polars(lf: pl.LazyFrame, n_levels: int = 5) -> pl.LazyFrame:
    """
    Trim orderbook LazyFrame to keep only top n levels.
    If n_levels=0, calculates simple mid price (bid_1 + ask_1) / 2 and drops all orderbook columns.
    """
    if n_levels == 0:
        # Calculate mid price and drop all orderbook columns
        return lf.with_columns([
            ((pl.col('bid_1_px') + pl.col('ask_1_px')) / 2).alias('mid_price')
        ]).select([
            col for col in lf.collect_schema().names()
            if not any(col.startswith(f'{side}_') for side in ['ask', 'bid'])
        ] + ['mid_price'])
    
    # Trim to n levels
    cols_to_keep = [
        col for col in lf.collect_schema().names()
        if not any(col.startswith(f'{side}_') for side in ['ask', 'bid']) or
        (col.split('_')[1].isdigit() and int(col.split('_')[1]) <= n_levels)
    ]
    return lf.select(cols_to_keep)

def bin_orderbook_polars(lf: pl.LazyFrame, freq: str) -> pl.LazyFrame:
    """
    Bin orderbook timestamps and keep most recent entry per time bin per symbol.
    Adds a 'time_bin' column with the bin timestamp. Leaves timeMs and exchTimeMs unchanged.
    freq: Polars duration string (e.g., '5m', '1m', '30s', '100ms')
    """
    # Convert timeMs to datetime for grouping
    return (lf
        .with_columns(pl.col('timeMs').cast(pl.Datetime('ms')).alias('_ts'))
        .sort('_ts')
        .group_by_dynamic('_ts', every=freq, by='symbol', closed='right', label='right')
        .agg([pl.all().last()])
        .with_columns(
            pl.col('_ts').alias('time_bin')  # Keep as datetime
        )
        .drop('_ts')
        .sort(['symbol', 'timeMs'])
    )
