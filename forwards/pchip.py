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
        symbols: Instrument symbols ('SWAP', 'BTC-USD-250905.OK', etc.)
    """
    timeMs: int
    T_nodes: np.ndarray
    ln_F_bid_nodes: np.ndarray
    ln_F_ask_nodes: np.ndarray
    symbols: list[str]
    
    @classmethod
    def from_polars(cls, df: pl.DataFrame):
        """
        Create PCHIPCurve(s) from a Polars DataFrame.
        
        Args:
            df: Polars DataFrame with columns: timeMs, T, ln_F_bid, ln_F_ask, symbol, source
            
        Returns:
            - Single PCHIPCurve if df contains one timeMs
            - List[PCHIPCurve] if df contains multiple timeMs values
            
        Examples:
            >>> # Single curve from filtered snapshot
            >>> curve = PCHIPCurve.from_polars(df.filter(pl.col('timeMs') == t))
            
            >>> # Multiple curves from full dataset
            >>> curves = PCHIPCurve.from_polars(df_pchip)
        """
        if df.is_empty():
            raise ValueError("Cannot create PCHIPCurve from empty DataFrame")
        
        # Get unique timestamps
        unique_times = df['timeMs'].unique().sort().to_list()
        
        if len(unique_times) == 1:
            # Single curve
            return cls(
                timeMs=int(df['timeMs'][0]),
                T_nodes=df['T'].to_numpy(),
                ln_F_bid_nodes=df['ln_F_bid'].to_numpy(),
                ln_F_ask_nodes=df['ln_F_ask'].to_numpy(),
                symbols=df['symbol'].to_list(),
            )
        else:
            # Multiple curves
            curves = []
            for t in unique_times:
                df_t = df.filter(pl.col('timeMs') == t)
                curves.append(cls(
                    timeMs=int(t),
                    T_nodes=df_t['T'].to_numpy(),
                    ln_F_bid_nodes=df_t['ln_F_bid'].to_numpy(),
                    ln_F_ask_nodes=df_t['ln_F_ask'].to_numpy(),
                    symbols=df_t['symbol'].to_list(),
                ))
            return curves
    
    def to_polars(self) -> pl.DataFrame:
        """Convert to Polars DataFrame for storage."""
        return pl.DataFrame({
            "timeMs": [self.timeMs] * len(self.T_nodes),
            "T": self.T_nodes,
            "ln_F_bid": self.ln_F_bid_nodes,
            "ln_F_ask": self.ln_F_ask_nodes,
            "symbol": self.symbols,
        })


def fit_pchip_curve(
    T_pillars: np.ndarray,
    F_bid_pillars: np.ndarray,
    F_ask_pillars: np.ndarray,
    symbols: list[str],
    timeMs: int,
) -> PCHIPCurve:
    """
    Fit PCHIP curve on log-forwards.
    
    Expects sorted, concatenated data (swap + futures) with LOG PRICES.
    Use concat_swap_and_pillars() to prepare data.
    
    Args:
        T_pillars: Time-to-maturity (sorted, T=0 at start)
        F_bid_pillars: Log bid prices (ln F)
        F_ask_pillars: Log ask prices (ln F)
        symbols: Instrument symbols (['SWAP', 'BTC-USD-250905.OK', ...])
        timeMs: Timestamp in milliseconds
        
    Returns:
        PCHIPCurve with fitted nodes
    """
    # Remove any duplicates in T (keep first occurrence)
    T_nodes, unique_idx = np.unique(T_pillars, return_index=True)
    ln_F_bid_nodes = F_bid_pillars[unique_idx]
    ln_F_ask_nodes = F_ask_pillars[unique_idx]
    symbols_nodes = [symbols[i] for i in unique_idx]
    
    return PCHIPCurve(
        timeMs=timeMs,
        T_nodes=T_nodes,
        ln_F_bid_nodes=ln_F_bid_nodes,
        ln_F_ask_nodes=ln_F_ask_nodes,
        symbols=symbols_nodes,
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
    Time-aware EWMA smoothing of pillar log-forwards.
    
    Uses continuous-time parameterization: α(Δt) = exp(-Δt/τ)
    This ensures frame-rate invariance across 1m, 5m, irregular binning.
    
    Tracks smoothed log-forward values for each (symbol, T) combination.
    """
    
    def __init__(self, tau_minutes: float = 5.0):
        """
        Initialize time-aware EWMA state.
        
        Args:
            tau_minutes: Time constant in minutes (higher = smoother, more lag)
                        Typical values: 3-10 minutes for crypto
                        Half-life = tau * ln(2) ≈ 0.693 * tau
        """
        self.tau_seconds = tau_minutes * 60.0  # Convert to seconds for precision
        self.state: Dict[str, Tuple[int, float, float, float]] = {}  
        # symbol -> (last_timeMs, T, ln_F_bid, ln_F_ask)
    
    def update(self, curve: PCHIPCurve) -> PCHIPCurve:
        """
        Update EWMA state with time-aware smoothing.
        
        For each pillar:
            α = exp(-Δt/τ)
            ŷ_smooth = α * ŷ_prev + (1-α) * y_obs
        
        This is frame-rate invariant: same behavior at 1m, 5m, or irregular times.
        
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
                # Compute actual elapsed time
                last_timeMs, T_prev, ln_bid_prev, ln_ask_prev = self.state[symbol]
                dt_seconds = (curve.timeMs - last_timeMs) / 1000.0
                
                # Time-aware EWMA weight: α = exp(-Δt/τ)
                alpha = np.exp(-dt_seconds / self.tau_seconds)
                
                # Smooth
                ln_bid_smooth = alpha * ln_bid_prev + (1 - alpha) * ln_bid_obs
                ln_ask_smooth = alpha * ln_ask_prev + (1 - alpha) * ln_ask_obs
            else:
                # Initialize with observation
                ln_bid_smooth = ln_bid_obs
                ln_ask_smooth = ln_ask_obs
            
            # Update state with timestamp
            self.state[symbol] = (curve.timeMs, T, ln_bid_smooth, ln_ask_smooth)
            smoothed_ln_bid.append(ln_bid_smooth)
            smoothed_ln_ask.append(ln_ask_smooth)
        
        return PCHIPCurve(
            timeMs=curve.timeMs,
            T_nodes=curve.T_nodes,
            ln_F_bid_nodes=np.array(smoothed_ln_bid),
            ln_F_ask_nodes=np.array(smoothed_ln_ask),
            symbols=curve.symbols,
        )
    
    def reset(self):
        """Clear EWMA state."""
        self.state.clear()


def ewma_smooth(
    curves: list[PCHIPCurve],
    tau_minutes: float = 5.0,
) -> list[PCHIPCurve]:
    """
    Apply time-aware EWMA smoothing to sequence of curves.
    
    Uses α(Δt) = exp(-Δt/τ) for frame-rate invariance.
    
    Args:
        curves: List of PCHIPCurve objects sorted by time
        tau_minutes: Time constant in minutes (typical: 3-10 for crypto)
                    Half-life = tau * ln(2) ≈ 0.693 * tau
        
    Returns:
        List of smoothed curves
        
    Examples:
        >>> curves = [fit_pchip_curve(...) for snapshot in snapshots]
        >>> smoothed = ewma_smooth(curves, tau_minutes=5.0)
    """
    state = EWMAState(tau_minutes=tau_minutes)
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

