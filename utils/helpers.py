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


def trim_orderbook(df: pd.DataFrame, n_levels: int = 5):
    # Keep non-orderbook columns and orderbook columns up to n_levels
    cols_to_keep = [
        col for col in df.columns
        if not any(col.startswith(f'{side}_') for side in ['ask', 'bid']) or
        (col.split('_')[1].isdigit() and int(col.split('_')[1]) <= n_levels)
    ]
    return df[cols_to_keep]