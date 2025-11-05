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
        F_ref_bid: Reference forward for bid curve
        F_ref_ask: Reference forward for ask curve
    """
    timeMs: int
    beta0: float
    beta1: float
    beta2: float
    lambda_ns: float
    F_ref_bid: float
    F_ref_ask: float
    
    def to_polars(self) -> pl.DataFrame:
        """Convert to Polars DataFrame for storage."""
        return pl.DataFrame({
            "timeMs": [self.timeMs],
            "beta0": [self.beta0],
            "beta1": [self.beta1],
            "beta2": [self.beta2],
            "lambda_ns": [self.lambda_ns],
            "F_ref_bid": [self.F_ref_bid],
            "F_ref_ask": [self.F_ref_ask],
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
    F_ref = state.F_ref_bid if use_bid else state.F_ref_ask
    
    integral = integrate_ns_carry(
        T_target,
        state.beta0,
        state.beta1,
        state.beta2,
        state.lambda_ns,
    )
    
    return F_ref * np.exp(integral)


class KalmanNSFilter:
    """
    Kalman filter for Nelson-Siegel carry curve estimation.
    
    State: θ = [β0, β1, β2]
    Transition: θ_t = A·θ_{t-1} + η_t, η_t ~ N(0, Q)
    Observation: y_i = H_i·θ + ε_i, ε_i ~ N(0, R_i)
    
    where y_i = ln F_i - ln F_ref is the log-forward relative to reference.
    """
    
    def __init__(
        self,
        lambda_ns: float = 0.1,
        process_noise_scale: float = 1e-4,
        ar1_coef: float = 0.99,
    ):
        """
        Initialize Kalman filter for NS carry estimation.
        
        Args:
            lambda_ns: Shape parameter (fixed, typically 0.05-0.2)
            process_noise_scale: Scale for process noise covariance Q
            ar1_coef: AR(1) coefficient for state transition (close to 1 for persistence)
        """
        self.lambda_ns = lambda_ns
        self.process_noise_scale = process_noise_scale
        
        # State dimension
        self.n_states = 3
        
        # State transition matrix (AR(1) per factor)
        self.A = np.eye(3) * ar1_coef
        
        # Process noise covariance
        self.Q = np.eye(3) * process_noise_scale
        
        # Initial state
        self.x = np.array([0.0, 0.0, 0.0])  # Start with zero carry
        self.P = np.eye(3) * 0.01  # Initial covariance
    
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
        T_pillars: np.ndarray,
        ln_F_obs: np.ndarray,
        ln_F_ref: float,
        measurement_variances: np.ndarray,
    ) -> np.ndarray:
        """
        Kalman filter update step.
        
        Args:
            T_pillars: Time-to-maturity for observed contracts
            ln_F_obs: Log-forward prices (observed)
            ln_F_ref: Log-reference price (typically perpetual swap)
            measurement_variances: Measurement noise variance for each observation
            
        Returns:
            Updated state estimate θ = [β0, β1, β2]
        """
        # Predict
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q
        
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
        self.P = np.eye(3) * 0.01


def kalman_filter(
    snapshots: list[dict],
    lambda_ns: float = 0.1,
    process_noise_scale: float = 1e-4,
    ar1_coef: float = 0.99,
    spread_to_variance_scale: float = 1.0,
) -> list[NSCarryState]:
    """
    Apply Kalman filter to sequence of orderbook snapshots.
    
    Args:
        snapshots: List of dicts with keys:
            - 'timeMs': Timestamp
            - 'T_pillars': Array of maturities
            - 'F_bid_pillars': Bid prices
            - 'F_ask_pillars': Ask prices
            - 'rel_spreads': Relative bid-ask spreads (for measurement noise)
            - 'F_ref_bid': Reference bid (swap)
            - 'F_ref_ask': Reference ask (swap)
        lambda_ns: Shape parameter
        process_noise_scale: Process noise scale
        ar1_coef: AR(1) coefficient
        spread_to_variance_scale: Scale to convert relative spread to measurement variance
        
    Returns:
        List of NSCarryState objects
    """
    filter_bid = KalmanNSFilter(lambda_ns, process_noise_scale, ar1_coef)
    filter_ask = KalmanNSFilter(lambda_ns, process_noise_scale, ar1_coef)
    
    states = []
    
    for snap in snapshots:
        timeMs = snap['timeMs']
        T_pillars = snap['T_pillars']
        F_bid_pillars = snap['F_bid_pillars']
        F_ask_pillars = snap['F_ask_pillars']
        rel_spreads = snap['rel_spreads']
        F_ref_bid = snap['F_ref_bid']
        F_ref_ask = snap['F_ref_ask']
        
        # Measurement noise from relative spreads (wider spread = more uncertain)
        # Variance in log-price space: rel_spread^2
        measurement_vars = (rel_spreads ** 2) * spread_to_variance_scale
        measurement_vars = np.maximum(measurement_vars, 1e-8)  # Floor to avoid numerical issues
        
        # Filter bid curve
        ln_F_bid_obs = np.log(F_bid_pillars)
        ln_F_ref_bid = np.log(F_ref_bid)
        theta_bid = filter_bid.update(T_pillars, ln_F_bid_obs, ln_F_ref_bid, measurement_vars)
        
        # Filter ask curve (independent)
        ln_F_ask_obs = np.log(F_ask_pillars)
        ln_F_ref_ask = np.log(F_ref_ask)
        theta_ask = filter_ask.update(T_pillars, ln_F_ask_obs, ln_F_ref_ask, measurement_vars)
        
        # Average the parameters (or store separately)
        # For simplicity, we'll average bid/ask factors
        beta0 = (theta_bid[0] + theta_ask[0]) / 2
        beta1 = (theta_bid[1] + theta_ask[1]) / 2
        beta2 = (theta_bid[2] + theta_ask[2]) / 2
        
        state = NSCarryState(
            timeMs=timeMs,
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
            lambda_ns=lambda_ns,
            F_ref_bid=F_ref_bid,
            F_ref_ask=F_ref_ask,
        )
        states.append(state)
    
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
    
    dfs = [state.to_polars() for state in states]
    return pl.concat(dfs)

