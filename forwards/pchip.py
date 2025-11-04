"""
PCHIP-based forward curve fitting with soft anchoring and EWMA smoothing.

Implements piecewise cubic Hermite interpolation on log-forwards with:
- Soft constraint at T=0 using perpetual swap as anchor
- EWMA smoothing across snapshots to reduce flicker
- Reconstruction of forward prices at arbitrary maturities
"""

import numpy as np
import polars as pl
from typing import Optional, Dict, Tuple
from scipy.interpolate import PchipInterpolator
from dataclasses import dataclass


@dataclass
class PCHIPCurve:
    """
    Container for PCHIP curve parameters.
    
    Attributes:
        timeMs: Timestamp in milliseconds
        T_nodes: Array of time-to-maturity nodes (years)
        ln_F_bid_nodes: Log-forward bid values at nodes
        ln_F_ask_nodes: Log-forward ask values at nodes
        symbols: Instrument symbols corresponding to nodes
        source: Source type for each node ('swap', 'observed', 'interpolated')
    """
    timeMs: int
    T_nodes: np.ndarray
    ln_F_bid_nodes: np.ndarray
    ln_F_ask_nodes: np.ndarray
    symbols: list[str]
    source: list[str]
    
    def to_polars(self) -> pl.DataFrame:
        """Convert to Polars DataFrame for storage."""
        return pl.DataFrame({
            "timeMs": [self.timeMs] * len(self.T_nodes),
            "T": self.T_nodes,
            "ln_F_bid": self.ln_F_bid_nodes,
            "ln_F_ask": self.ln_F_ask_nodes,
            "F_bid": np.exp(self.ln_F_bid_nodes),
            "F_ask": np.exp(self.ln_F_ask_nodes),
            "symbol": self.symbols,
            "source": self.source,
        })


def fit_pchip_curve(
    T_pillars: np.ndarray,
    F_bid_pillars: np.ndarray,
    F_ask_pillars: np.ndarray,
    symbols: list[str],
    timeMs: int,
    T_swap: float = 0.0,
    F_bid_swap: Optional[float] = None,
    F_ask_swap: Optional[float] = None,
    w0_anchor: float = 10.0,
    weights: Optional[np.ndarray] = None,
) -> PCHIPCurve:
    """
    Fit PCHIP curve on log-forwards with soft swap anchor.
    
    Uses weighted least-squares to add a soft constraint at T=0, then fits
    PCHIP interpolator on the resulting node values.
    
    Args:
        T_pillars: Time-to-maturity for futures contracts (years)
        F_bid_pillars: Bid prices for futures
        F_ask_pillars: Ask prices for futures
        symbols: Instrument symbols for each pillar
        timeMs: Timestamp in milliseconds
        T_swap: Time-to-maturity for swap (typically 0.0 for perpetual)
        F_bid_swap: Swap bid price (if None, no anchor)
        F_ask_swap: Swap ask price (if None, no anchor)
        w0_anchor: Weight for swap anchor constraint (higher = stronger anchor)
        weights: Optional weights for each pillar (default: equal weights)
        
    Returns:
        PCHIPCurve with fitted nodes
        
    Notes:
        - PCHIP guarantees monotonicity if data is monotonic
        - Soft anchoring prevents unrealistic spot-futures divergence
        - Works in log-space to ensure positive forwards
    """
    # Prepare arrays
    if weights is None:
        weights = np.ones(len(T_pillars))
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Add swap anchor if provided
    if F_bid_swap is not None and F_ask_swap is not None:
        T_all = np.concatenate([[T_swap], T_pillars])
        F_bid_all = np.concatenate([[F_bid_swap], F_bid_pillars])
        F_ask_all = np.concatenate([[F_ask_swap], F_ask_pillars])
        symbols_all = ["SWAP"] + list(symbols)
        source_all = ["swap"] + ["observed"] * len(T_pillars)
        
        # Scale anchor weight relative to pillar weights
        anchor_weight = w0_anchor / len(T_pillars)
        weights_all = np.concatenate([[anchor_weight], weights * (1 - anchor_weight)])
    else:
        T_all = T_pillars
        F_bid_all = F_bid_pillars
        F_ask_all = F_ask_pillars
        symbols_all = list(symbols)
        source_all = ["observed"] * len(T_pillars)
        weights_all = weights
    
    # Sort by T (PCHIP requires sorted x)
    sort_idx = np.argsort(T_all)
    T_sorted = T_all[sort_idx]
    F_bid_sorted = F_bid_all[sort_idx]
    F_ask_sorted = F_ask_all[sort_idx]
    symbols_sorted = [symbols_all[i] for i in sort_idx]
    source_sorted = [source_all[i] for i in sort_idx]
    
    # Work in log space
    ln_F_bid = np.log(F_bid_sorted)
    ln_F_ask = np.log(F_ask_sorted)
    
    # Remove any duplicates in T (keep first occurrence)
    unique_T, unique_idx = np.unique(T_sorted, return_index=True)
    T_nodes = unique_T
    ln_F_bid_nodes = ln_F_bid[unique_idx]
    ln_F_ask_nodes = ln_F_ask[unique_idx]
    symbols_nodes = [symbols_sorted[i] for i in unique_idx]
    source_nodes = [source_sorted[i] for i in unique_idx]
    
    return PCHIPCurve(
        timeMs=timeMs,
        T_nodes=T_nodes,
        ln_F_bid_nodes=ln_F_bid_nodes,
        ln_F_ask_nodes=ln_F_ask_nodes,
        symbols=symbols_nodes,
        source=source_nodes,
    )


def reconstruct_forward(
    curve: PCHIPCurve,
    T_target: float | np.ndarray,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """
    Reconstruct forward prices at target maturity from PCHIP curve.
    
    Args:
        curve: Fitted PCHIPCurve
        T_target: Target time-to-maturity (scalar or array)
        
    Returns:
        (F_bid, F_ask) tuple at target maturity
        
    Examples:
        >>> curve = fit_pchip_curve(...)
        >>> F_bid, F_ask = reconstruct_forward(curve, T_target=0.5)
    """
    # Create PCHIP interpolators
    pchip_bid = PchipInterpolator(curve.T_nodes, curve.ln_F_bid_nodes, extrapolate=True)
    pchip_ask = PchipInterpolator(curve.T_nodes, curve.ln_F_ask_nodes, extrapolate=True)
    
    # Interpolate in log-space, then exponentiate
    ln_F_bid_target = pchip_bid(T_target)
    ln_F_ask_target = pchip_ask(T_target)
    
    F_bid_target = np.exp(ln_F_bid_target)
    F_ask_target = np.exp(ln_F_ask_target)
    
    return F_bid_target, F_ask_target


class EWMAState:
    """
    State container for EWMA smoothing of pillar values over time.
    
    Tracks smoothed log-forward values for each (symbol, T) combination.
    """
    
    def __init__(self, lambda_ewma: float = 0.8):
        """
        Initialize EWMA state.
        
        Args:
            lambda_ewma: Smoothing parameter (0.7-0.9 typical)
                        Higher = smoother but more lag
        """
        self.lambda_ewma = lambda_ewma
        self.state: Dict[str, Tuple[float, float, float]] = {}  # symbol -> (T, ln_F_bid, ln_F_ask)
    
    def update(self, curve: PCHIPCurve) -> PCHIPCurve:
        """
        Update EWMA state and return smoothed curve.
        
        Args:
            curve: New observed curve
            
        Returns:
            Smoothed PCHIPCurve
        """
        smoothed_ln_bid = []
        smoothed_ln_ask = []
        
        for i, symbol in enumerate(curve.symbols):
            T = curve.T_nodes[i]
            ln_bid_obs = curve.ln_F_bid_nodes[i]
            ln_ask_obs = curve.ln_F_ask_nodes[i]
            
            if symbol in self.state:
                # EWMA update
                T_prev, ln_bid_prev, ln_ask_prev = self.state[symbol]
                ln_bid_smooth = self.lambda_ewma * ln_bid_prev + (1 - self.lambda_ewma) * ln_bid_obs
                ln_ask_smooth = self.lambda_ewma * ln_ask_prev + (1 - self.lambda_ewma) * ln_ask_obs
            else:
                # Initialize with observation
                ln_bid_smooth = ln_bid_obs
                ln_ask_smooth = ln_ask_obs
            
            self.state[symbol] = (T, ln_bid_smooth, ln_ask_smooth)
            smoothed_ln_bid.append(ln_bid_smooth)
            smoothed_ln_ask.append(ln_ask_smooth)
        
        return PCHIPCurve(
            timeMs=curve.timeMs,
            T_nodes=curve.T_nodes,
            ln_F_bid_nodes=np.array(smoothed_ln_bid),
            ln_F_ask_nodes=np.array(smoothed_ln_ask),
            symbols=curve.symbols,
            source=curve.source,
        )
    
    def reset(self):
        """Clear EWMA state."""
        self.state.clear()


def ewma_smooth(
    curves: list[PCHIPCurve],
    lambda_ewma: float = 0.8,
) -> list[PCHIPCurve]:
    """
    Apply EWMA smoothing to sequence of curves.
    
    Args:
        curves: List of PCHIPCurve objects sorted by time
        lambda_ewma: Smoothing parameter (0.7-0.9 typical)
        
    Returns:
        List of smoothed curves
        
    Examples:
        >>> curves = [fit_pchip_curve(...) for snapshot in snapshots]
        >>> smoothed = ewma_smooth(curves, lambda_ewma=0.8)
    """
    state = EWMAState(lambda_ewma=lambda_ewma)
    smoothed_curves = []
    
    for curve in curves:
        smoothed_curve = state.update(curve)
        smoothed_curves.append(smoothed_curve)
    
    return smoothed_curves


def curves_to_polars(curves: list[PCHIPCurve]) -> pl.DataFrame:
    """
    Convert list of PCHIPCurve objects to single Polars DataFrame.
    
    Args:
        curves: List of fitted curves
        
    Returns:
        Combined DataFrame with all curve nodes
    """
    if not curves:
        return pl.DataFrame()
    
    dfs = [curve.to_polars() for curve in curves]
    return pl.concat(dfs)

