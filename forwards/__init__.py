"""
Forward curve modeling for crypto derivatives.

Exchange-agnostic implementations of curve fitting, evaluation, and utilities.
Supports PCHIP interpolation with EWMA smoothing and Kalman-filtered Nelson-Siegel carry models.
"""

__version__ = "0.1.0"

from .pchip import (
    fit_pchip_curve,
    ewma_smooth,
    reconstruct_forward,
    PCHIPCurve,
    curves_to_polars,
)
from .kalman_ns import (
    nelson_siegel_carry,
    integrate_ns_carry,
    kalman_filter,
    reconstruct_ns_forward,
    NSCarryState,
    KalmanNSFilter,
    states_to_polars,
)
from .evaluation import (
    wmae_pillar_fit,
    leave_one_expiry_out,
    temporal_smoothness,
    calendar_spread_check,
    diagnostics,
    evaluate_curve_snapshot,
    EvaluationMetrics,
)
from .utils import (
    datetime_to_year_fraction,
    compute_weights,
    apply_early_roll_filter,
    parse_futures_expiry,
    add_expiry_and_ttm,
)

__all__ = [
    # PCHIP
    "fit_pchip_curve",
    "ewma_smooth",
    "reconstruct_forward",
    "PCHIPCurve",
    "curves_to_polars",
    # Kalman NS
    "nelson_siegel_carry",
    "integrate_ns_carry",
    "kalman_filter",
    "reconstruct_ns_forward",
    "NSCarryState",
    "KalmanNSFilter",
    "states_to_polars",
    # Evaluation
    "wmae_pillar_fit",
    "leave_one_expiry_out",
    "temporal_smoothness",
    "calendar_spread_check",
    "diagnostics",
    "evaluate_curve_snapshot",
    "EvaluationMetrics",
    # Utils
    "datetime_to_year_fraction",
    "compute_weights",
    "apply_early_roll_filter",
    "parse_futures_expiry",
    "add_expiry_and_ttm",
]

