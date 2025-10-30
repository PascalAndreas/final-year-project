import pandas as pd
import numpy as np
from typing import Optional, Tuple


def compute_segment_carry(futures_df: pd.DataFrame, swap_bid: float, swap_ask: float, swap_time: pd.Timestamp, 
                          min_time_to_expiry: Optional[float] = None) -> pd.DataFrame:
    """Compute piecewise-constant carry rates for upper/lower forward bounds using perpetual swap anchor.
    min_time_to_expiry: Minimum time to expiry in hours (e.g., 4.0). Contracts closer to expiry are dropped."""
    sorted_futures = futures_df.sort_values('expiry').copy()
    sorted_futures['T'] = ((sorted_futures['expiry'] - swap_time).dt.total_seconds() / (365.25 * 24 * 3600))
    
    # Filter out near-expiry contracts if specified
    if min_time_to_expiry is not None:
        min_T = min_time_to_expiry / (365.25 * 24)  # Convert hours to years
        sorted_futures = sorted_futures[sorted_futures['T'] >= min_T].copy()
    
    if len(sorted_futures) == 0:
        return pd.DataFrame()
    
    segments = []
    T_start, expiry_start = 0.0, swap_time
    F_lower_start, F_upper_start = swap_bid, swap_ask
    segment_num = 1
    
    for idx, row in sorted_futures.iterrows():
        T_end, expiry_end = row['T'], row['expiry']
        F_lower_end, F_upper_end = row['bid_1_px'], row['ask_1_px']
        
        # Compute carry rates for both bounds
        if T_end > T_start and F_lower_start > 0 and F_lower_end > 0:
            carry_lower = (np.log(F_lower_end) - np.log(F_lower_start)) / (T_end - T_start)
        else:
            carry_lower = 0.0
            
        if T_end > T_start and F_upper_start > 0 and F_upper_end > 0:
            carry_upper = (np.log(F_upper_end) - np.log(F_upper_start)) / (T_end - T_start)
        else:
            carry_upper = 0.0
        
        prev_symbol = None if idx == 0 else sorted_futures.iloc[sorted_futures.index.get_loc(idx) - 1]['symbol']
        
        segments.append({
            'segment_num': segment_num,
            'T_start': T_start, 'T_end': T_end,
            'F_lower_start': F_lower_start, 'F_lower_end': F_lower_end,
            'F_upper_start': F_upper_start, 'F_upper_end': F_upper_end,
            'carry_lower': carry_lower, 'carry_upper': carry_upper,
            'expiry_start': expiry_start, 'expiry_end': expiry_end,
            'symbol_start': 'SWAP' if T_start == 0 else prev_symbol,
            'symbol_end': row['symbol']
        })
        
        T_start, expiry_start = T_end, expiry_end
        F_lower_start, F_upper_start = F_lower_end, F_upper_end
        segment_num += 1
    
    return pd.DataFrame(segments)


def interpolate_forward(T_target: float, segments_df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """Interpolate forward bounds at T_target using piecewise-constant carry. Returns (lower, upper)."""
    if T_target < 0:
        return None
    
    for _, segment in segments_df.iterrows():
        if segment['T_start'] <= T_target <= segment['T_end']:
            F_lower = segment['F_lower_start'] * np.exp(segment['carry_lower'] * (T_target - segment['T_start']))
            F_upper = segment['F_upper_start'] * np.exp(segment['carry_upper'] * (T_target - segment['T_start']))
            return F_lower, F_upper
    
    if T_target > segments_df['T_end'].max():
        last = segments_df.iloc[-1]
        F_lower = last['F_lower_start'] * np.exp(last['carry_lower'] * (T_target - last['T_start']))
        F_upper = last['F_upper_start'] * np.exp(last['carry_upper'] * (T_target - last['T_start']))
        return F_lower, F_upper
    
    return None


def compute_continuous_forward_curve(
    futures_df: pd.DataFrame,
    swap_bid: float,
    swap_ask: float,
    swap_time: pd.Timestamp,
    T_grid: Optional[np.ndarray] = None,
    min_time_to_expiry: Optional[float] = None
) -> pd.DataFrame:
    """Compute forward curve bounds using piecewise-constant carry interpolation with perpetual swap anchor.
    min_time_to_expiry: Minimum time to expiry in hours (e.g., 4.0). Contracts closer to expiry are dropped."""
    segments_df = compute_segment_carry(futures_df, swap_bid, swap_ask, swap_time, min_time_to_expiry)
    
    T_values = np.concatenate([[0], segments_df['T_end'].values]) if T_grid is None else T_grid
    forward_curve = []
    
    for T in T_values:
        if T == 0:
            forward_curve.append({'T': T, 'F_lower': swap_bid, 'F_upper': swap_ask,
                                 'expiry': swap_time, 'source': 'swap'})
        else:
            F_bounds = interpolate_forward(T, segments_df)
            if F_bounds is not None:
                is_observed = any(np.isclose(T, segments_df['T_end'], atol=1e-6))
                forward_curve.append({
                    'T': T, 'F_lower': F_bounds[0], 'F_upper': F_bounds[1],
                    'expiry': swap_time + pd.Timedelta(days=T * 365.25),
                    'source': 'observed' if is_observed else 'interpolated'
                })
    
    return pd.DataFrame(forward_curve)


def build_forward_surface(
    swap_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    T_grid: Optional[np.ndarray] = None,
    min_time_to_expiry: Optional[float] = None
) -> pd.DataFrame:
    """Build forward surface bounds over time using perpetual swap anchor.
    min_time_to_expiry: Minimum time to expiry in hours (e.g., 4.0). Contracts closer to expiry are dropped."""
    all_curves = []
    
    for time_bin, swap_row in swap_df.iterrows():
        swap_bid, swap_ask = swap_row['bid_1_px'], swap_row['ask_1_px']
        swap_time = swap_row['time_bin']
        
        futures_at_time = futures_df[futures_df['time_bin'] == swap_time].copy()
        if len(futures_at_time) == 0:
            continue
        
        curve = compute_continuous_forward_curve(futures_at_time, swap_bid, swap_ask, swap_time, T_grid, min_time_to_expiry)
        curve['time_bin'] = swap_time
        all_curves.append(curve)
    
    return pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()


def add_mid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Add mid-price columns to forward curve or segment DataFrame containing F_lower/F_upper."""
    df = df.copy()
    if 'F_lower' in df.columns and 'F_upper' in df.columns:
        df['F_mid'] = (df['F_lower'] + df['F_upper']) / 2
    if 'F_lower_start' in df.columns and 'F_upper_start' in df.columns:
        df['F_mid_start'] = (df['F_lower_start'] + df['F_upper_start']) / 2
    if 'F_lower_end' in df.columns and 'F_upper_end' in df.columns:
        df['F_mid_end'] = (df['F_lower_end'] + df['F_upper_end']) / 2
    if 'carry_lower' in df.columns and 'carry_upper' in df.columns:
        df['carry_mid'] = (df['carry_lower'] + df['carry_upper']) / 2
    return df

