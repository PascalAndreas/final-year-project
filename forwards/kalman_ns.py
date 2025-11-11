"""
Kalman-filtered Nelson-Siegel carry curve for forward pricing.

Implements a state-space model where the carry curve evolves smoothly over time:
- State: Nelson-Siegel factors θ = [β0, β1, β2]
- Transition: AR(1) dynamics with process noise
- Observation: Log-forward prices from futures orderbook
- Measurement noise: Based on bid-ask spreads

The carry curve is: c(T; θ) = β0 + β1·e^(-λT) + β2·T·e^(-λT)
Log-forward: ln F(T) = ln F_ref + ∫₀ᵀ c(u; θ) du (closed form)
"""

import numpy as np
import polars as pl
from typing import Tuple, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm


@dataclass
class NSCarryState:
    """
    Nelson-Siegel carry state parameters.
    
    Attributes:
        timeMs: Timestamp in milliseconds
        beta0: Level factor (long-term carry)
        beta1: Slope factor (short-term carry deviation)
        beta2: Curvature factor (medium-term hump)
        lambda_ns: Shape parameter (fixed or estimated)
        ln_F_ref_bid: Log reference forward for bid curve
        ln_F_ref_ask: Log reference forward for ask curve
    """
    timeMs: int
    beta0: float
    beta1: float
    beta2: float
    lambda_ns: float
    ln_F_ref_bid: float
    ln_F_ref_ask: float
    
    @classmethod
    def from_polars(cls, df: pl.DataFrame):
        """
        Create NSCarryState(s) from a Polars DataFrame.
        
        Args:
            df: Polars DataFrame with columns: timeMs, beta0, beta1, beta2, lambda_ns, F_ref_bid, F_ref_ask
            
        Returns:
            - Single NSCarryState if df contains one row
            - List[NSCarryState] if df contains multiple rows
            
        Examples:
            >>> # Single state from filtered snapshot
            >>> state = NSCarryState.from_polars(df.filter(pl.col('timeMs') == t))
            
            >>> # Multiple states from full dataset
            >>> states = NSCarryState.from_polars(df_kalman)
        """
        if df.is_empty():
            raise ValueError("Cannot create NSCarryState from empty DataFrame")
        
        if len(df) == 1:
            # Single state
            return cls(
                timeMs=int(df['timeMs'][0]),
                beta0=float(df['beta0'][0]),
                beta1=float(df['beta1'][0]),
                beta2=float(df['beta2'][0]),
                lambda_ns=float(df['lambda_ns'][0]),
                ln_F_ref_bid=float(df['ln_F_ref_bid'][0]),
                ln_F_ref_ask=float(df['ln_F_ref_ask'][0]),
            )
        else:
            # Multiple states
            states = []
            for i in range(len(df)):
                states.append(cls(
                    timeMs=int(df['timeMs'][i]),
                    beta0=float(df['beta0'][i]),
                    beta1=float(df['beta1'][i]),
                    beta2=float(df['beta2'][i]),
                    lambda_ns=float(df['lambda_ns'][i]),
                    ln_F_ref_bid=float(df['ln_F_ref_bid'][i]),
                    ln_F_ref_ask=float(df['ln_F_ref_ask'][i]),
                ))
            return states
    
    def to_polars(self) -> pl.DataFrame:
        """Convert to Polars DataFrame for storage."""
        return pl.DataFrame({
            "timeMs": [self.timeMs],
            "beta0": [self.beta0],
            "beta1": [self.beta1],
            "beta2": [self.beta2],
            "lambda_ns": [self.lambda_ns],
            "ln_F_ref_bid": [self.ln_F_ref_bid],
            "ln_F_ref_ask": [self.ln_F_ref_ask],
        })


def nelson_siegel_carry(T: np.ndarray, beta0: float, beta1: float, beta2: float, lambda_ns: float) -> np.ndarray:
    """
    Compute Nelson-Siegel carry curve.
    
    c(T) = β0 + β1·e^(-λT) + β2·T·e^(-λT)
    
    Args:
        T: Time-to-maturity array (years)
        beta0: Level factor
        beta1: Slope factor
        beta2: Curvature factor
        lambda_ns: Shape parameter (typically 0.05-0.2)
        
    Returns:
        Carry rates at each T
    """
    exp_term = np.exp(-lambda_ns * T)
    return beta0 + beta1 * exp_term + beta2 * T * exp_term


def integrate_ns_carry(T: np.ndarray, beta0: float, beta1: float, beta2: float, lambda_ns: float) -> np.ndarray:
    """
    Compute integral of Nelson-Siegel carry curve from 0 to T.
    
    ∫₀ᵀ c(u) du where c(u) = β0 + β1·e^(-λu) + β2·u·e^(-λu)
    
    Closed form:
    = β0·T + β1·(1 - e^(-λT))/λ + β2·(λT - 1 + e^(-λT))/λ²
    
    Args:
        T: Time-to-maturity array (years)
        beta0: Level factor
        beta1: Slope factor
        beta2: Curvature factor
        lambda_ns: Shape parameter
        
    Returns:
        Integrated carry from 0 to each T
    """
    exp_term = np.exp(-lambda_ns * T)
    
    term0 = beta0 * T
    term1 = beta1 * (1 - exp_term) / lambda_ns
    term2 = beta2 * (lambda_ns * T - 1 + exp_term) / (lambda_ns ** 2)
    
    return term0 + term1 + term2


def reconstruct_ns_forward(
    state: NSCarryState,
    T_target: np.ndarray | float,
    use_bid: bool = True,
) -> np.ndarray | float:
    """
    Reconstruct forward price at target maturity from NS state.
    
    ln F(T) = ln F_ref + ∫₀ᵀ c(u; θ) du
    F(T) = F_ref · exp(∫₀ᵀ c(u; θ) du)
    
    Args:
        state: NS carry state parameters
        T_target: Target time-to-maturity (years)
        use_bid: If True, use bid reference; otherwise use ask
        
    Returns:
        Forward price(s) at target maturity
    """
    ln_F_ref = state.ln_F_ref_bid if use_bid else state.ln_F_ref_ask
    
    integral = integrate_ns_carry(
        T_target,
        state.beta0,
        state.beta1,
        state.beta2,
        state.lambda_ns,
    )
    
    return np.exp(ln_F_ref + integral)


class KalmanNSFilter:
    """
    Time-aware Kalman filter for Nelson-Siegel carry curve estimation.
    
    Models carry factors as independent Ornstein-Uhlenbeck processes:
        dβ_k = -(1/τ_k)·β_k·dt + σ_k·dW_k
    
    Uses exact OU discretization for frame-rate invariance:
        A_k = exp(-Δt/τ_k)
        Q_k = (σ_k²·τ_k/2)·(1 - exp(-2Δt/τ_k))
    
    State: θ = [β0, β1, β2]
    Observation: y_i = H_i·θ + ε_i, where y_i = ln F_i - ln F_ref
    """
    
    def __init__(
        self,
        lambda_ns: float = 1.0,
        tau_minutes: np.ndarray = None,
        sigma_per_sqrt_day: np.ndarray = None,
    ):
        """
        Initialize time-aware Kalman filter for NS carry estimation.
        
        Args:
            lambda_ns: Shape parameter (fixed, typically 0.5-2.0/year for crypto)
            tau_minutes: Time constants [τ0, τ1, τ2] in minutes for each factor
                        Default: [2880, 7200, 14400] (2d, 5d, 10d)
                        Higher = more persistent
            sigma_per_sqrt_day: Volatility [σ0, σ1, σ2] per sqrt(day) for each factor
                               Default: [0.01, 0.01, 0.01]
                               Controls innovation size
        """
        self.lambda_ns = lambda_ns
        self.n_states = 3
        
        # Default time constants: front factor faster, back factor slower
        if tau_minutes is None:
            tau_minutes = np.array([2880.0, 7200.0, 14400.0])  # 2d, 5d, 10d
        self.tau_seconds = tau_minutes * 60.0  # Convert to seconds
        
        # Default volatilities
        if sigma_per_sqrt_day is None:
            sigma_per_sqrt_day = np.array([0.01, 0.01, 0.01])
        self.sigma = sigma_per_sqrt_day / np.sqrt(86400.0)  # Convert to per sqrt(second)
        
        # Initial state: zero carry
        self.x = np.array([0.0, 0.0, 0.0])
        
        # Initial covariance: stationary covariance of OU process
        # P_stationary = σ_k²·τ_k / 2
        self.P = np.diag(self.sigma ** 2 * self.tau_seconds / 2)
        
        # Track last update time for Δt computation
        self.last_update_time = None
    
    def _build_observation_matrix(self, T_pillars: np.ndarray) -> np.ndarray:
        """
        Build observation matrix H where each row corresponds to one pillar.
        
        H_i = [∫₀ᵀⁱ 1 du, ∫₀ᵀⁱ e^(-λu) du, ∫₀ᵀⁱ u·e^(-λu) du]
        
        Args:
            T_pillars: Time-to-maturity for each observed futures contract
            
        Returns:
            H matrix of shape (n_obs, 3)
        """
        n_obs = len(T_pillars)
        H = np.zeros((n_obs, 3))
        
        for i, T in enumerate(T_pillars):
            exp_term = np.exp(-self.lambda_ns * T)
            
            # Integrals of each NS basis function from 0 to T
            H[i, 0] = T  # ∫ 1 du = T
            H[i, 1] = (1 - exp_term) / self.lambda_ns  # ∫ e^(-λu) du
            H[i, 2] = (self.lambda_ns * T - 1 + exp_term) / (self.lambda_ns ** 2)  # ∫ u·e^(-λu) du
        
        return H
    
    def update(
        self,
        timeMs: int,
        T_pillars: np.ndarray,
        ln_F_obs: np.ndarray,
        ln_F_ref: float,
        measurement_variances: np.ndarray,
    ) -> np.ndarray:
        """
        Time-aware Kalman filter update step.
        
        Discretizes OU dynamics based on actual Δt:
            A_k = exp(-Δt/τ_k)
            Q_k = (σ_k²·τ_k/2)·(1 - exp(-2Δt/τ_k))
        
        Args:
            timeMs: Current timestamp in milliseconds
            T_pillars: Time-to-maturity for observed contracts
            ln_F_obs: Log-forward prices (observed)
            ln_F_ref: Log-reference price (typically perpetual swap)
            measurement_variances: Measurement noise variance for each observation
            
        Returns:
            Updated state estimate θ = [β0, β1, β2]
        """
        # Compute time step for OU discretization
        if self.last_update_time is None:
            dt = 0.0  # First update: no prediction step
        else:
            dt = (timeMs - self.last_update_time) / 1000.0  # seconds
        
        self.last_update_time = timeMs
        
        # Exact OU discretization for each factor
        if dt > 0:
            # A_k = exp(-Δt/τ_k)
            A_diag = np.exp(-dt / self.tau_seconds)
            A = np.diag(A_diag)
            
            # Q_k = (σ_k²·τ_k/2)·(1 - exp(-2Δt/τ_k))
            Q_diag = (self.sigma ** 2 * self.tau_seconds / 2) * (1 - np.exp(-2 * dt / self.tau_seconds))
            Q = np.diag(Q_diag)
        else:
            # No time passed: identity transition, zero noise
            A = np.eye(self.n_states)
            Q = np.zeros((self.n_states, self.n_states))
        
        # Predict
        x_pred = A @ self.x
        P_pred = A @ self.P @ A.T + Q
        
        # Build observation model
        H = self._build_observation_matrix(T_pillars)
        R = np.diag(measurement_variances)
        
        # Innovation
        y_obs = ln_F_obs - ln_F_ref  # Observed log-carry integral
        y_pred = H @ x_pred  # Predicted log-carry integral
        innovation = y_obs - y_pred
        
        # Innovation covariance
        S = H @ P_pred @ H.T + R
        
        # Kalman gain
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse if S is singular
            K = P_pred @ H.T @ np.linalg.pinv(S)
        
        # Update
        self.x = x_pred + K @ innovation
        self.P = (np.eye(self.n_states) - K @ H) @ P_pred
        
        return self.x.copy()
    
    def reset(self):
        """Reset filter to initial state."""
        self.x = np.array([0.0, 0.0, 0.0])
        self.P = np.diag(self.sigma ** 2 * self.tau_seconds / 2)  # Stationary covariance
        self.last_update_time = None


def kalman_filter(
    snapshots: list[dict],
    lambda_ns: float = 1.0,
    tau_minutes: np.ndarray = None,
    sigma_per_sqrt_day: np.ndarray = None,
    kappa_spread: float = 0.5,
    R_min: float = 1e-8,
    R_max: float = 1e-2,
    progress: bool = False,
) -> list[NSCarryState]:
    """
    Apply time-aware Kalman filter to sequence of orderbook snapshots.
    
    Uses exact OU discretization and spread-based measurement noise for
    frame-rate invariance and adaptive noise handling.
    
    Args:
        snapshots: List of dicts with keys:
            - 'timeMs': Timestamp in milliseconds
            - 'T': Array of maturities (including T=0 for swap)
            - 'ln_F_bid': Log bid prices
            - 'ln_F_ask': Log ask prices
            - 'rel_spreads': Relative bid-ask spreads (for measurement noise)
        lambda_ns: Shape parameter (0.5-2.0/year typical for crypto)
        tau_minutes: Time constants [τ0, τ1, τ2] in minutes (default: [2d, 5d, 10d])
        sigma_per_sqrt_day: Volatilities [σ0, σ1, σ2] per sqrt(day) (default: [0.01, 0.01, 0.01])
        kappa_spread: Scale factor for spread-based measurement noise (0.5-1.0 typical)
        R_min: Minimum measurement variance (floor to avoid singularity)
        R_max: Maximum measurement variance (cap for very wide spreads)
        
    Returns:
        List of NSCarryState objects with time-aware filtering
    """
    filter_bid = KalmanNSFilter(lambda_ns, tau_minutes, sigma_per_sqrt_day)
    filter_ask = KalmanNSFilter(lambda_ns, tau_minutes, sigma_per_sqrt_day)
    
    states = []
    
    iterator = snapshots
    progress_bar = None
    if progress:
        progress_bar = tqdm(snapshots, desc="Kalman filter")
        iterator = progress_bar
    
    for snap in iterator:
        timeMs = snap['timeMs']
        T_pillars = snap['T']
        ln_F_bid_pillars = snap['ln_F_bid']
        ln_F_ask_pillars = snap['ln_F_ask']
        rel_spreads = snap['rel_spreads']
        
        # Reference prices are at T=0 (first element)
        ln_F_ref_bid = ln_F_bid_pillars[0]
        ln_F_ref_ask = ln_F_ask_pillars[0]
        
        # Spread-based measurement noise in log-price space
        # R_j = (κ · spread_j)²  (already in relative terms)
        measurement_vars = (kappa_spread * rel_spreads) ** 2
        
        # Clip to [R_min, R_max] for numerical stability
        measurement_vars = np.clip(measurement_vars, R_min, R_max)
        
        # Filter bid curve
        theta_bid = filter_bid.update(timeMs, T_pillars, ln_F_bid_pillars, ln_F_ref_bid, measurement_vars)
        
        # Filter ask curve (independent)
        theta_ask = filter_ask.update(timeMs, T_pillars, ln_F_ask_pillars, ln_F_ref_ask, measurement_vars)
        
        # Average bid/ask factors for single curve representation
        beta0 = (theta_bid[0] + theta_ask[0]) / 2
        beta1 = (theta_bid[1] + theta_ask[1]) / 2
        beta2 = (theta_bid[2] + theta_ask[2]) / 2
        
        state = NSCarryState(
            timeMs=timeMs,
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
            lambda_ns=lambda_ns,
            ln_F_ref_bid=ln_F_ref_bid,
            ln_F_ref_ask=ln_F_ref_ask,
        )
        states.append(state)
    
    if progress_bar is not None:
        progress_bar.close()
    return states


def states_to_polars(states: list[NSCarryState]) -> pl.DataFrame:
    """
    Convert list of NSCarryState objects to single Polars DataFrame.
    
    Args:
        states: List of Kalman filter states
        
    Returns:
        Combined DataFrame with all states
    """
    if not states:
        return pl.DataFrame()
    
    time_values = np.fromiter((state.timeMs for state in states), dtype=np.int64, count=len(states))
    beta0_values = np.fromiter((state.beta0 for state in states), dtype=np.float64, count=len(states))
    beta1_values = np.fromiter((state.beta1 for state in states), dtype=np.float64, count=len(states))
    beta2_values = np.fromiter((state.beta2 for state in states), dtype=np.float64, count=len(states))
    lambda_values = np.fromiter((state.lambda_ns for state in states), dtype=np.float64, count=len(states))
    ref_bid_values = np.fromiter((state.ln_F_ref_bid for state in states), dtype=np.float64, count=len(states))
    ref_ask_values = np.fromiter((state.ln_F_ref_ask for state in states), dtype=np.float64, count=len(states))
    
    return pl.DataFrame({
        "timeMs": time_values,
        "beta0": beta0_values,
        "beta1": beta1_values,
        "beta2": beta2_values,
        "lambda_ns": lambda_values,
        "ln_F_ref_bid": ref_bid_values,
        "ln_F_ref_ask": ref_ask_values,
    })
