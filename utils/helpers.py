from datetime import datetime
import pandas as pd

def parse_option_instrument(instrument_name: str):
    parts = instrument_name.split('-')
    
    if len(parts) != 5:
        raise ValueError(f"Invalid instrument name format: {instrument_name}")
    
    underlying, quote, expiry_str, strike_str, option_type = parts

    return f"{parts[0]}-{parts[1]}", datetime.strptime(parts[2], '%y%m%d'), int(parts[3]), parts[4]

def get_option_combos(df: pd.DataFrame):
    # Extract expiry/strike pairs from valid instruments
    combos_df = pd.DataFrame([
        parse_option_instrument(inst)[1:3] 
        for inst in df.iloc[:, 0].unique()
        if len(inst.split('-')) == 5
    ], columns=['expiry', 'strike'])
    
    return combos_df.drop_duplicates().sort_values(['expiry', 'strike']).reset_index(drop=True)

def standardize_orderbook_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Standardize orderbook column names from FUTURES format to OPTIONS format.
    E.g. askPx1 -> ask_1_px, bidSz2 -> bid_2_sz
    Also adds a symbol column from the filename.
    """
    # Add symbol column from filename
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
            rename_map[col] = f"{side}_{num_str}_{col_type}"
            
    return df.rename(columns=rename_map)

def trim_orderbook(df: pd.DataFrame, n_levels: int = 5):
    """
    Trim orderbook DataFrame to keep only top n levels.
    Expects OPTIONS format column names (ask_1_px, bid_1_px, etc.)
    """
    cols_to_keep = [
        col for col in df.columns
        if not any(col.startswith(f'{side}_') for side in ['ask', 'bid']) or
        (col.split('_')[1].isdigit() and int(col.split('_')[1]) <= n_levels)
    ]
    return df[cols_to_keep]
