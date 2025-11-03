import pandas as pd
from datetime import datetime

def parse_option_name(instrument_name: str):
    # Remove file extension if present (e.g., 'BTC-USD-250627-200000-C.OK' -> 'BTC-USD-250627-200000-C')
    instrument = instrument_name.split('.')[0]
    parts = instrument.split('-')
    if len(parts) != 5:
        raise ValueError(f"Invalid instrument name format: {instrument}")
    
    # Parse date and set expiry time to 08:00 UTC (timezone-naive for pandas compatibility)
    expiry_datetime = datetime.strptime(parts[2], '%y%m%d').replace(hour=8, minute=0, second=0, microsecond=0)
    
    return f"{parts[0]}-{parts[1]}", expiry_datetime, int(parts[3]), parts[4].upper()

def parse_future_name(instrument_name: str):
    # Remove file extension if present (e.g., 'BTC-USD-250131.OK' -> 'BTC-USD-250131')
    instrument = instrument_name.split('.')[0]
    parts = instrument.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid instrument name format: {instrument}")
    
    # Parse date and set expiry time to 08:00 UTC (timezone-naive for pandas compatibility)
    expiry_datetime = datetime.strptime(parts[2], '%y%m%d').replace(hour=8, minute=0, second=0, microsecond=0)
    
    return f"{parts[0]}-{parts[1]}", expiry_datetime

def candle_to_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert candlestick OHLC data to bid/ask bounds format for forward pricing."""
    df = df.copy()
    df['bid_1_px'] = df['low']
    df['ask_1_px'] = df['high']
    df['time_bin'] = pd.to_datetime(df['open_time'], unit='ms')
    df['timestamp'] = df['open_time']
    if 'instrument_name' in df.columns:
        df['symbol'] = df['instrument_name']
    return df