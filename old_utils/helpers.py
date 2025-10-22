import numpy as np

def moneyness(K, F):
    """Calculate forward moneyness (K/F)."""
    return K / F if F > 0 else np.nan

def parse_option_instrument(instrument_name):
    """Parse option instrument name to extract expiry, strike, and type."""
    parts = instrument_name.split('-')
    if len(parts) != 4:
        return None, None, None
    return parts[1], int(parts[2]), parts[3]