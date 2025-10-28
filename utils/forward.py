import pandas as pd
import numpy as np
from typing import Optional


def compute_segment_carry(futures_df: pd.DataFrame, spot_mid: float, spot_time: pd.Timestamp) -> pd.DataFrame:
    """
    Compute piecewise-constant carry rates between listed futures expiries.
    
    Algorithm:
    - Sort futures by maturity: 0 < T1 < ... < Tn
    - Anchor with spot S as F0 at T=0
    - Compute segment carry: c_j = [ln F_{j+1} - ln F_j] / (T_{j+1} - T_j)
    
    Args:
        futures_df: DataFrame with columns ['symbol', 'mid_price', 'expiry']
        spot_mid: Spot mid price (used as F0)
        spot_time: Current time for the spot price
    
    Returns:
        DataFrame with columns ['segment', 'T_start', 'T_end', 'F_start', 'F_end', 
                                'carry_rate', 'expiry_start', 'expiry_end']
    """
    # Sort by expiry
    sorted_futures = futures_df.sort_values('expiry').copy()
    
    # Calculate time to maturity in years for each future
    sorted_futures['T'] = (
        (sorted_futures['expiry'] - spot_time).dt.total_seconds() / (365.25 * 24 * 3600)
    )
    
    # Build segments
    segments = []
    
    # Segment 0: Spot to first future
    T_start = 0.0
    F_start = spot_mid
    expiry_start = spot_time
    
    for idx, row in sorted_futures.iterrows():
        T_end = row['T']
        F_end = row['mid_price']
        expiry_end = row['expiry']
        
        # Compute carry rate: c = [ln(F_end) - ln(F_start)] / (T_end - T_start)
        if T_end > T_start and F_start > 0 and F_end > 0:
            carry_rate = (np.log(F_end) - np.log(F_start)) / (T_end - T_start)
        else:
            carry_rate = 0.0
        
        # Get previous symbol safely
        prev_symbol = None
        if idx > 0:
            prev_idx = sorted_futures.index[sorted_futures.index.get_loc(idx) - 1]
            prev_symbol = sorted_futures.loc[prev_idx, 'symbol']
        
        segments.append({
            'T_start': T_start,
            'T_end': T_end,
            'F_start': F_start,
            'F_end': F_end,
            'carry_rate': carry_rate,
            'carry_rate_pct': carry_rate * 100,
            'expiry_start': expiry_start,
            'expiry_end': expiry_end,
            'symbol_start': 'SPOT' if T_start == 0 else prev_symbol,
            'symbol_end': row['symbol']
        })
        
        # Update for next segment
        T_start = T_end
        F_start = F_end
        expiry_start = expiry_end
    
    return pd.DataFrame(segments)


def interpolate_forward(T_target: float, segments_df: pd.DataFrame) -> Optional[float]:
    """
    Interpolate forward price at target maturity T_target using piecewise-constant carry.
    
    Formula: F(T*) = F_j · exp(c_j · (T* - T_j))
    where segment j contains T* ∈ [T_j, T_{j+1}]
    
    Args:
        T_target: Target time to maturity in years
        segments_df: DataFrame from compute_segment_carry
    
    Returns:
        Interpolated forward price, or None if extrapolation beyond last segment
    """
    if T_target < 0:
        return None
    
    # Find the segment containing T_target
    for _, segment in segments_df.iterrows():
        if segment['T_start'] <= T_target <= segment['T_end']:
            # Interpolate within this segment
            F_interpolated = segment['F_start'] * np.exp(
                segment['carry_rate'] * (T_target - segment['T_start'])
            )
            return F_interpolated
    
    # Extrapolation beyond last segment (hold carry constant)
    if T_target > segments_df['T_end'].max():
        last_segment = segments_df.iloc[-1]
        F_interpolated = last_segment['F_start'] * np.exp(
            last_segment['carry_rate'] * (T_target - last_segment['T_start'])
        )
        return F_interpolated
    
    return None


def compute_continuous_forward_curve(
    futures_df: pd.DataFrame,
    spot_mid: float,
    spot_time: pd.Timestamp,
    T_grid: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute the complete forward curve using piecewise-constant carry interpolation.
    
    Args:
        futures_df: DataFrame with columns ['symbol', 'mid_price', 'expiry']
        spot_mid: Spot mid price
        spot_time: Current time for the spot price
        T_grid: Optional array of maturities (in years) to evaluate. 
                If None, uses the observed futures maturities.
    
    Returns:
        DataFrame with columns ['T', 'F', 'expiry', 'source'] where source is 
        'observed' or 'interpolated'
    """
    # Compute segment carry rates
    segments_df = compute_segment_carry(futures_df, spot_mid, spot_time)
    
    if T_grid is None:
        # Use observed maturities
        T_values = np.concatenate([[0], segments_df['T_end'].values])
    else:
        T_values = T_grid
    
    # Interpolate forward prices
    forward_curve = []
    
    for T in T_values:
        if T == 0:
            forward_curve.append({
                'T': T,
                'F': spot_mid,
                'expiry': spot_time,
                'source': 'spot'
            })
        else:
            F_interp = interpolate_forward(T, segments_df)
            if F_interp is not None:
                # Determine if this is an observed or interpolated point
                is_observed = any(np.isclose(T, segments_df['T_end'], atol=1e-6))
                
                forward_curve.append({
                    'T': T,
                    'F': F_interp,
                    'expiry': spot_time + pd.Timedelta(days=T * 365.25),
                    'source': 'observed' if is_observed else 'interpolated'
                })
    
    return pd.DataFrame(forward_curve)


def build_forward_surface(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    T_grid: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Build a forward surface over time by computing forward curves at each timestamp.
    
    Args:
        spot_df: DataFrame with columns ['time_bin', 'mid_price']
        futures_df: DataFrame with columns ['time_bin', 'symbol', 'mid_price', 'expiry']
        T_grid: Optional array of maturities (in years) to evaluate
    
    Returns:
        DataFrame with columns ['time_bin', 'T', 'F', 'expiry', 'source']
    """
    all_curves = []
    
    # Group futures by time_bin
    for time_bin, spot_row in spot_df.iterrows():
        spot_mid = spot_row['mid_price']
        spot_time = spot_row['time_bin']
        
        # Get futures for this timestamp
        futures_at_time = futures_df[futures_df['time_bin'] == spot_time].copy()
        
        if len(futures_at_time) == 0:
            continue
        
        # Compute forward curve for this timestamp
        curve = compute_continuous_forward_curve(
            futures_at_time,
            spot_mid,
            spot_time,
            T_grid
        )
        
        curve['time_bin'] = spot_time
        all_curves.append(curve)
    
    if not all_curves:
        return pd.DataFrame()
    
    return pd.concat(all_curves, ignore_index=True)

