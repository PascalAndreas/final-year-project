"""
Utility functions for forward curve modeling.
"""

import numpy as np
import polars as pl
from datetime import datetime, timezone
from typing import Optional


def datetime_to_year_fraction(
    dt: datetime | pl.Expr, reference: datetime | pl.Expr
) -> float | pl.Expr:
    """
    Convert datetime to year fraction relative to reference time.
    
    Args:
        dt: Target datetime (or Polars expression)
        reference: Reference datetime (or Polars expression)
        
    Returns:
        Year fraction T (or Polars expression)
        
    Examples:
        >>> ref = datetime(2025, 1, 1, tzinfo=timezone.utc)
        >>> target = datetime(2025, 7, 1, tzinfo=timezone.utc)
        >>> datetime_to_year_fraction(target, ref)
        0.4986...
    """
    if isinstance(dt, pl.Expr) and isinstance(reference, pl.Expr):
        # Polars expression: compute difference in seconds, convert to years
        seconds_per_year = 365.25 * 24 * 3600
        return (dt - reference).dt.total_seconds() / seconds_per_year
    elif isinstance(dt, datetime) and isinstance(reference, datetime):
        # Python datetime: direct computation
        seconds_diff = (dt - reference).total_seconds()
        return seconds_diff / (365.25 * 24 * 3600)
    else:
        raise TypeError("Both dt and reference must be same type (datetime or pl.Expr)")


def compute_weights(
    spreads: np.ndarray | pl.Expr,
    max_weight_ratio: float = 100.0,
    min_spread_bps: float = 0.1,
) -> np.ndarray | pl.Expr:
    """
    Compute inverse-spread-squared weights with capping.
    
    Weights are proportional to 1/spread² but capped to avoid extreme domination
    by very tight spreads.
    
    Args:
        spreads: Array of bid-ask spreads (absolute or relative)
        max_weight_ratio: Maximum ratio between largest and smallest weight
        min_spread_bps: Minimum spread in bps to avoid division by zero
        
    Returns:
        Normalized weights (sum to 1)
        
    Examples:
        >>> spreads = np.array([0.01, 0.02, 0.05])
        >>> weights = compute_weights(spreads)
        >>> weights.sum()
        1.0
    """
    if isinstance(spreads, pl.Expr):
        # Polars expression version
        spread_floor = pl.lit(min_spread_bps / 10000.0)  # Convert bps to decimal
        raw_weights = 1.0 / pl.max_horizontal(spreads, spread_floor).pow(2)
        
        # Cap weights
        w_max = raw_weights.max()
        w_min = w_max / max_weight_ratio
        capped_weights = pl.max_horizontal(raw_weights, w_min)
        
        # Normalize
        return capped_weights / capped_weights.sum()
    else:
        # NumPy array version
        spread_floor = min_spread_bps / 10000.0
        spreads_safe = np.maximum(spreads, spread_floor)
        raw_weights = 1.0 / (spreads_safe ** 2)
        
        # Cap weights
        w_max = raw_weights.max()
        w_min = w_max / max_weight_ratio
        capped_weights = np.maximum(raw_weights, w_min)
        
        # Normalize
        return capped_weights / capped_weights.sum()


def apply_early_roll_filter(
    df: pl.DataFrame,
    current_time_col: str = "timeMs",
    expiry_col: str = "expiry",
    min_time_to_expiry_hours: float = 2.0,
) -> pl.DataFrame:
    """
    Filter out contracts close to expiry (early-roll rule).
    
    Args:
        df: DataFrame with futures contracts
        current_time_col: Column name for current timestamp (milliseconds)
        expiry_col: Column name for expiry timestamp (datetime or milliseconds)
        min_time_to_expiry_hours: Minimum hours to expiry (drop contracts closer than this)
        
    Returns:
        Filtered DataFrame
        
    Examples:
        >>> df = pl.DataFrame({
        ...     "timeMs": [1000000000] * 3,
        ...     "expiry": [1000000000 + i * 3600000 for i in [1, 3, 10]],
        ...     "symbol": ["A", "B", "C"]
        ... })
        >>> filtered = apply_early_roll_filter(df, min_time_to_expiry_hours=2.0)
        >>> len(filtered)
        2
    """
    min_seconds = min_time_to_expiry_hours * 3600
    
    # Convert to datetime if needed and compute time to expiry
    if df.schema[expiry_col] == pl.Datetime:
        expiry_ms = pl.col(expiry_col).dt.epoch("ms")
    else:
        # Assume it's already in milliseconds
        expiry_ms = pl.col(expiry_col)
    
    time_to_expiry_seconds = (expiry_ms - pl.col(current_time_col)) / 1000.0
    
    return df.filter(time_to_expiry_seconds >= min_seconds)