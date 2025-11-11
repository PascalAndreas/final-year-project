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
    loeo_error,
    temporal_smoothness,
    calendar_spread_check,
    diagnostics,
    evaluate_curve_snapshot,
    EvaluationMetrics,
    evaluate_forward_parity,
)
from .parity import compute_option_parity_table, summarize_option_parity
from .plotting import (
    plot_pchip_snapshot,
    plot_kalman_snapshot,
    plot_ns_factors,
    plot_forward_tracking,
    plot_error_distribution,
    plot_spread_analysis,
    create_comparison_figure,
)
from .data import load_matched_options

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
    "loeo_error",
    "temporal_smoothness",
    "calendar_spread_check",
    "diagnostics",
    "evaluate_curve_snapshot",
    "EvaluationMetrics",
    "compute_option_parity_table",
    "summarize_option_parity",
    "evaluate_forward_parity",
    # Data helpers
    "load_matched_options",
    # Plotting
    "plot_pchip_snapshot",
    "plot_kalman_snapshot",
    "plot_ns_factors",
    "plot_forward_tracking",
    "plot_error_distribution",
    "plot_spread_analysis",
    "create_comparison_figure",
]
