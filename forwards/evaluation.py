"""
Evaluation metrics for forward curve quality.

Implements various metrics to assess:
- Pillar fit quality (WMAE in bps)
- Cross-validation (leave-one-expiry-out)
- Temporal smoothness vs tracking
- Calendar spread consistency
- Diagnostic quality checks
"""

import numpy as np
import polars as pl
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation results."""
    metric_name: str
    value: float
    details: Optional[Dict[str, Any]] = None
    
    def to_polars(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        data = {
            "metric_name": [self.metric_name],
            "value": [self.value],
        }
        if self.details:
            for key, val in self.details.items():
                if isinstance(val, (int, float, str)):
                    data[key] = [val]
        return pl.DataFrame(data)


def wmae_pillar_fit(
    T_pillars: np.ndarray,
    F_obs: np.ndarray,
    F_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> EvaluationMetrics:
    """
    Weighted Mean Absolute Error of pillar fit in basis points.
    
    E = mean_i |ln F̂_i - ln F_i| · w_i · 10000
    
    Args:
        T_pillars: Time-to-maturity for each pillar
        F_obs: Observed forward prices
        F_pred: Predicted forward prices
        weights: Optional weights for each pillar (default: equal)
        
    Returns:
        EvaluationMetrics with WMAE in bps
        
    Examples:
        >>> wmae = wmae_pillar_fit(T, F_obs, F_pred, weights)
        >>> print(f"WMAE: {wmae.value:.2f} bps")
    """
    if weights is None:
        weights = np.ones(len(T_pillars)) / len(T_pillars)
    else:
        weights = weights / weights.sum()
    
    # Compute log errors
    ln_errors = np.abs(np.log(F_pred) - np.log(F_obs))
    
    # Convert to bps and weight
    errors_bps = ln_errors * 10000
    wmae_bps = np.sum(errors_bps * weights)
    
    # Additional statistics
    mae_bps = np.mean(errors_bps)
    max_error_bps = np.max(errors_bps)
    
    return EvaluationMetrics(
        metric_name="wmae_pillar_fit",
        value=wmae_bps,
        details={
            "mae_bps": mae_bps,
            "max_error_bps": max_error_bps,
            "n_pillars": len(T_pillars),
        }
    )


def leave_one_expiry_out(
    T_pillars: np.ndarray,
    F_bid_pillars: np.ndarray,
    F_ask_pillars: np.ndarray,
    symbols: list[str],
    fit_func: Callable,
    reconstruct_func: Callable,
    weights: Optional[np.ndarray] = None,
    **fit_kwargs,
) -> pl.DataFrame:
    """
    Leave-one-expiry-out cross-validation.
    
    For each pillar:
    1. Remove it from the dataset
    2. Fit curve on remaining pillars
    3. Reconstruct forward at removed maturity
    4. Compute error
    
    Args:
        T_pillars: Time-to-maturity for each pillar
        F_bid_pillars: Bid prices
        F_ask_pillars: Ask prices
        symbols: Instrument symbols
        fit_func: Curve fitting function (e.g., fit_pchip_curve)
        reconstruct_func: Reconstruction function (e.g., reconstruct_forward)
        weights: Optional weights
        **fit_kwargs: Additional arguments for fit_func
        
    Returns:
        DataFrame with per-pillar LOEO errors
    """
    n_pillars = len(T_pillars)
    results = []
    
    for i in range(n_pillars):
        # Leave out pillar i
        mask = np.ones(n_pillars, dtype=bool)
        mask[i] = False
        
        T_train = T_pillars[mask]
        F_bid_train = F_bid_pillars[mask]
        F_ask_train = F_ask_pillars[mask]
        symbols_train = [s for j, s in enumerate(symbols) if mask[j]]
        
        if weights is not None:
            weights_train = weights[mask]
            weights_train = weights_train / weights_train.sum()
        else:
            weights_train = None
        
        # Fit on training set
        try:
            curve = fit_func(
                T_train, F_bid_train, F_ask_train, symbols_train,
                weights=weights_train,
                **fit_kwargs
            )
            
            # Reconstruct at held-out maturity
            F_bid_pred, F_ask_pred = reconstruct_func(curve, T_pillars[i])
            
            # Compute errors in bps
            error_bid_bps = np.abs(np.log(F_bid_pred) - np.log(F_bid_pillars[i])) * 10000
            error_ask_bps = np.abs(np.log(F_ask_pred) - np.log(F_ask_pillars[i])) * 10000
            error_mid_bps = (error_bid_bps + error_ask_bps) / 2
            
            results.append({
                "symbol": symbols[i],
                "T": T_pillars[i],
                "F_bid_obs": F_bid_pillars[i],
                "F_ask_obs": F_ask_pillars[i],
                "F_bid_pred": F_bid_pred,
                "F_ask_pred": F_ask_pred,
                "error_bid_bps": error_bid_bps,
                "error_ask_bps": error_ask_bps,
                "error_mid_bps": error_mid_bps,
                "success": True,
            })
        except Exception as e:
            results.append({
                "symbol": symbols[i],
                "T": T_pillars[i],
                "F_bid_obs": F_bid_pillars[i],
                "F_ask_obs": F_ask_pillars[i],
                "F_bid_pred": None,
                "F_ask_pred": None,
                "error_bid_bps": None,
                "error_ask_bps": None,
                "error_mid_bps": None,
                "success": False,
                "error": str(e),
            })
    
    return pl.DataFrame(results)


def temporal_smoothness(
    forward_series: pl.DataFrame,
    perp_series: pl.DataFrame,
    test_maturities: list[float],
    reconstruct_func: Callable,
) -> pl.DataFrame:
    """
    Measure temporal smoothness vs tracking quality.
    
    For each test maturity T*:
    - Smoothness: var(ΔF(T*)) / var(Δperp) - lower is smoother
    - Tracking: corr(ΔF(T*), Δperp) - shouldn't collapse
    
    Args:
        forward_series: DataFrame with forward curve parameters over time
        perp_series: DataFrame with perpetual swap prices over time
        test_maturities: List of maturities to test (years)
        reconstruct_func: Function to reconstruct F(T) from parameters
        
    Returns:
        DataFrame with smoothness metrics for each test maturity
    """
    # Align timestamps
    forward_series = forward_series.sort("timeMs")
    perp_series = perp_series.sort("timeMs")
    
    # Join on timestamp
    combined = forward_series.join(perp_series, on="timeMs", how="inner")
    
    if len(combined) < 2:
        return pl.DataFrame({
            "T_test": test_maturities,
            "smoothness_ratio": [None] * len(test_maturities),
            "tracking_corr": [None] * len(test_maturities),
        })
    
    # Compute perp changes
    perp_prices = combined["perp_mid"].to_numpy()
    delta_perp = np.diff(np.log(perp_prices))
    var_perp = np.var(delta_perp)
    
    results = []
    for T_test in test_maturities:
        # Reconstruct forwards at this maturity for all snapshots
        # Note: This assumes forward_series has the necessary data for reconstruction
        # Implementation depends on whether it's PCHIP nodes or NS parameters
        
        # Placeholder: user would need to provide appropriate reconstruction
        # For now, we'll create a simplified version
        
        try:
            # This is a simplified example - actual implementation depends on data format
            F_values = []
            for row in combined.iter_rows(named=True):
                # Extract parameters and reconstruct
                # This would call reconstruct_func with appropriate params
                pass
            
            if len(F_values) < 2:
                smoothness_ratio = None
                tracking_corr = None
            else:
                delta_F = np.diff(np.log(F_values))
                var_F = np.var(delta_F)
                smoothness_ratio = var_F / var_perp if var_perp > 0 else None
                
                # Correlation
                if len(delta_F) == len(delta_perp):
                    tracking_corr = np.corrcoef(delta_F, delta_perp)[0, 1]
                else:
                    tracking_corr = None
            
            results.append({
                "T_test": T_test,
                "smoothness_ratio": smoothness_ratio,
                "tracking_corr": tracking_corr,
            })
        except Exception:
            results.append({
                "T_test": T_test,
                "smoothness_ratio": None,
                "tracking_corr": None,
            })
    
    return pl.DataFrame(results)


def calendar_spread_check(
    T_pillars: np.ndarray,
    F_pred: np.ndarray,
    F_obs_spreads: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """
    Check calendar spread consistency.
    
    Model-implied spread: F̂(T_{j+1}) / F̂(T_j)
    Should match observed spreads within tolerance.
    
    Args:
        T_pillars: Sorted time-to-maturity array
        F_pred: Predicted forward prices at pillars
        F_obs_spreads: Optional observed calendar spreads for comparison
        
    Returns:
        DataFrame with calendar spread diagnostics
    """
    if len(T_pillars) < 2:
        return pl.DataFrame()
    
    # Model-implied spreads
    model_spreads = F_pred[1:] / F_pred[:-1]
    model_spread_bps = (np.log(model_spreads)) * 10000
    
    results = []
    for i in range(len(T_pillars) - 1):
        result = {
            "T_near": T_pillars[i],
            "T_far": T_pillars[i + 1],
            "F_near": F_pred[i],
            "F_far": F_pred[i + 1],
            "model_spread_ratio": model_spreads[i],
            "model_spread_bps": model_spread_bps[i],
        }
        
        if F_obs_spreads is not None and len(F_obs_spreads) > i:
            obs_spread_bps = np.log(F_obs_spreads[i]) * 10000
            result["obs_spread_bps"] = obs_spread_bps
            result["error_bps"] = model_spread_bps[i] - obs_spread_bps
        
        results.append(result)
    
    return pl.DataFrame(results)


def diagnostics(
    errors: np.ndarray,
    threshold_bps: float = 50.0,
) -> EvaluationMetrics:
    """
    Compute diagnostic statistics for curve quality.
    
    Args:
        errors: Array of errors in bps
        threshold_bps: Threshold for outlier detection
        
    Returns:
        EvaluationMetrics with diagnostic statistics
    """
    n_total = len(errors)
    n_outliers = np.sum(np.abs(errors) > threshold_bps)
    pct_outliers = (n_outliers / n_total) * 100 if n_total > 0 else 0
    
    # Spike detection (large changes)
    if len(errors) > 1:
        deltas = np.abs(np.diff(errors))
        n_spikes = np.sum(deltas > threshold_bps)
        pct_spikes = (n_spikes / (n_total - 1)) * 100 if n_total > 1 else 0
    else:
        n_spikes = 0
        pct_spikes = 0
    
    return EvaluationMetrics(
        metric_name="diagnostics",
        value=pct_outliers,
        details={
            "n_total": n_total,
            "n_outliers": n_outliers,
            "pct_outliers": pct_outliers,
            "n_spikes": n_spikes,
            "pct_spikes": pct_spikes,
            "threshold_bps": threshold_bps,
        }
    )


def evaluate_curve_snapshot(
    T_pillars: np.ndarray,
    F_bid_obs: np.ndarray,
    F_ask_obs: np.ndarray,
    F_bid_pred: np.ndarray,
    F_ask_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """
    Comprehensive evaluation for a single curve snapshot.
    
    Args:
        T_pillars: Time-to-maturity array
        F_bid_obs: Observed bid prices
        F_ask_obs: Observed ask prices
        F_bid_pred: Predicted bid prices
        F_ask_pred: Predicted ask prices
        weights: Optional pillar weights
        
    Returns:
        DataFrame with all evaluation metrics
    """
    metrics = []
    
    # Bid WMAE
    wmae_bid = wmae_pillar_fit(T_pillars, F_bid_obs, F_bid_pred, weights)
    metrics.append(wmae_bid.to_polars().with_columns(pl.lit("bid").alias("side")))
    
    # Ask WMAE
    wmae_ask = wmae_pillar_fit(T_pillars, F_ask_obs, F_ask_pred, weights)
    metrics.append(wmae_ask.to_polars().with_columns(pl.lit("ask").alias("side")))
    
    # Mid WMAE
    F_mid_obs = (F_bid_obs + F_ask_obs) / 2
    F_mid_pred = (F_bid_pred + F_ask_pred) / 2
    wmae_mid = wmae_pillar_fit(T_pillars, F_mid_obs, F_mid_pred, weights)
    metrics.append(wmae_mid.to_polars().with_columns(pl.lit("mid").alias("side")))
    
    # Calendar spreads
    cal_bid = calendar_spread_check(T_pillars, F_bid_pred)
    cal_ask = calendar_spread_check(T_pillars, F_ask_pred)
    
    return pl.concat(metrics)


def loeo_error(
    store,
    dates: list,
    forwards_recipe: Callable,
    recipe_kwargs: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Compute leave-one-expiry-out cross-validation error for forward curves.
    
    For each pillar at each timestamp:
        1. Rebuild curve with that pillar excluded (using drop_pillar_idx)
        2. Predict F(T_pillar) from fitted curve
        3. Compare to observed F_pillar
        4. Compute error in bps
    
    This function integrates with the recipe system and prepare_pillars to efficiently
    evaluate out-of-sample prediction performance.
    
    Args:
        store: OrderbookStore instance
        dates: List of dates to evaluate
        forwards_recipe: Recipe function (e.g., build_forwards_pchip or build_forwards_kalman)
        recipe_kwargs: Optional dict of kwargs to pass to recipe (e.g., binning, tau_ewma_minutes)
        
    Returns:
        DataFrame with columns:
            - timeMs: Observation timestamp
            - pillar_idx: Index of dropped pillar
            - symbol: Contract symbol (e.g., 'BTC-USD-251031.OK')
            - T: Time to maturity (years)
            - F_bid_obs, F_ask_obs: Observed forward prices
            - F_bid_pred, F_ask_pred: Predicted forward prices
            - error_bid_bps, error_ask_bps, error_mid_bps: Errors in basis points
            - success: Whether prediction succeeded
            
    Examples:
        >>> from functools import partial
        >>> from forwards.evaluation import loeo_error
        >>> from okx.recipes.forwards import build_forwards_pchip
        >>> 
        >>> # Evaluate PCHIP recipe
        >>> recipe = build_forwards_pchip
        >>> kwargs = {'binning': '5m', 'tau_ewma_minutes': 5.0}
        >>> df_loeo = loeo_error(store, dates, recipe, kwargs)
        >>> 
        >>> # Compute summary statistics
        >>> print(df_loeo.filter(pl.col('success'))['error_mid_bps'].describe())
    """
    if recipe_kwargs is None:
        recipe_kwargs = {}
    
    # Import here to avoid circular dependency
    from okx.recipes.forwards import prepare_pillars
    
    # First, get the full pillar data without dropping anything to determine n_pillars per snapshot
    inst_family = recipe_kwargs.get('inst_family', 'BTC-USD')
    binning = recipe_kwargs.get('binning', '5m')
    min_time_to_expiry_hours = recipe_kwargs.get('min_time_to_expiry_hours', 2.0)
    unique_times = recipe_kwargs.get('unique_times', None)
    cache_name_suffix = recipe_kwargs.get('cache_name_suffix', 'pillars')

    # Determine recipe type for reconstruction
    from functools import partial
    recipe_callable = forwards_recipe.func if isinstance(forwards_recipe, partial) else forwards_recipe
    recipe_name = recipe_callable.__name__.lower()
    if 'pchip' in recipe_name:
        recipe_kind = 'pchip'
    elif 'kalman' in recipe_name:
        recipe_kind = 'kalman'
    else:
        raise ValueError(f"Unsupported forwards recipe for LOEO evaluation: {recipe_callable.__name__}")
    
    # Get reference snapshots to determine number of pillars at each time
    pillars_lf = prepare_pillars(
        store, inst_family, dates, binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        cache_name_suffix=cache_name_suffix,
        drop_pillar_idx=None,
    )
    
    df_pillars = pillars_lf.collect()
    
    if df_pillars.is_empty():
        return pl.DataFrame()
    
    results = []
    
    # For each snapshot, determine the number of pillars
    for pillars_df in df_pillars.partition_by('timeMs', maintain_order=True):
        timeMs = int(pillars_df['timeMs'][0])
        n_pillars = len(pillars_df)

        if n_pillars <= 1:
            continue
        
        # Skip SWAP (idx 0) since it's the anchor
        for pillar_idx in range(1, n_pillars):
            # Get observed data for this pillar
            row = pillars_df.row(pillar_idx, named=True)
            symbol_held_out = row['symbol']
            T_held_out = row['T']
            F_bid_obs = np.exp(row['ln_bid_1_px'])
            F_ask_obs = np.exp(row['ln_ask_1_px'])
            
            try:
                # Build curve with this pillar dropped
                # Pass unique_times to only build for this specific timestamp
                recipe_kwargs_loeo = recipe_kwargs.copy()
                recipe_kwargs_loeo['drop_pillar_idx'] = pillar_idx
                recipe_kwargs_loeo['unique_times'] = [timeMs]
                recipe_kwargs_loeo['dates'] = dates
                
                # Remove cache_name_suffix - it's not a valid recipe parameter
                recipe_kwargs_loeo.pop('cache_name_suffix', None)
                
                # Build curve without EWMA smoothing to get single-snapshot prediction
                # (EWMA would require full time series context)
                if 'tau_ewma_minutes' in recipe_kwargs_loeo:
                    # For PCHIP, set tau very small to effectively disable smoothing
                    recipe_kwargs_loeo['tau_ewma_minutes'] = 0.001
                
                lf_pred = forwards_recipe(store, **recipe_kwargs_loeo)
                df_pred = lf_pred.collect()
                
                if df_pred.is_empty():
                    raise ValueError("Recipe returned empty DataFrame")
                
                # Get the curve for this timestamp
                df_curve = df_pred.filter(pl.col('timeMs') == timeMs)
                
                if df_curve.is_empty():
                    raise ValueError(f"No curve found for timeMs={timeMs}")
                
                # Reconstruct forward at held-out maturity using interpolation
                # Import PCHIPCurve for reconstruction
                if recipe_kind == 'pchip':
                    from forwards.pchip import PCHIPCurve, reconstruct_forward
                    
                    curve = PCHIPCurve(
                        timeMs=timeMs,
                        T_nodes=df_curve['T'].to_numpy(),
                        ln_F_bid_nodes=df_curve['ln_F_bid'].to_numpy(),
                        ln_F_ask_nodes=df_curve['ln_F_ask'].to_numpy(),
                        symbols=df_curve['symbol'].to_list(),
                    )
                    F_bid_pred, F_ask_pred = reconstruct_forward(curve, T_held_out)
                else:
                    from forwards.kalman_ns import NSCarryState, reconstruct_ns_forward
                    state = NSCarryState.from_polars(df_curve)
                    F_bid_pred = reconstruct_ns_forward(state, T_held_out, use_bid=True)
                    F_ask_pred = reconstruct_ns_forward(state, T_held_out, use_bid=False)
                
                # Compute errors in bps
                error_bid_bps = np.abs(np.log(F_bid_pred) - np.log(F_bid_obs)) * 10000
                error_ask_bps = np.abs(np.log(F_ask_pred) - np.log(F_ask_obs)) * 10000
                error_mid_bps = (error_bid_bps + error_ask_bps) / 2
                
                results.append({
                    'timeMs': timeMs,
                    'pillar_idx': pillar_idx,
                    'symbol': symbol_held_out,
                    'T': T_held_out,
                    'F_bid_obs': F_bid_obs,
                    'F_ask_obs': F_ask_obs,
                    'F_bid_pred': float(F_bid_pred),
                    'F_ask_pred': float(F_ask_pred),
                    'error_bid_bps': float(error_bid_bps),
                    'error_ask_bps': float(error_ask_bps),
                    'error_mid_bps': float(error_mid_bps),
                    'success': True,
                })
                
            except Exception as e:
                results.append({
                    'timeMs': timeMs,
                    'pillar_idx': pillar_idx,
                    'symbol': symbol_held_out,
                    'T': T_held_out,
                    'F_bid_obs': F_bid_obs,
                    'F_ask_obs': F_ask_obs,
                    'F_bid_pred': None,
                    'F_ask_pred': None,
                    'error_bid_bps': None,
                    'error_ask_bps': None,
                    'error_mid_bps': None,
                    'success': False,
                    'error': str(e),
                })
    
    return pl.DataFrame(results)
