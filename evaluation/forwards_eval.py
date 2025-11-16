"""
Evaluation metrics for forward curve quality.

Implements various metrics to assess:
- Put-call parity: compare fitted forwards to options-implied forwards
- Pillar fit quality: WMAE between fitted curve and pillar observations
- Leave-one-expiry-out (LOEO): Cross-validation by excluding each pillar
"""
from datetime import date
from typing import Optional

import numpy as np
import polars as pl
from tqdm import tqdm
from functools import partial

from okx.recipes.forwards import assign_forwards
from okx.recipes.pillars import prepare_pillars
from okx.recipes.options import prepare_options

# =============================================================================
# Options Parity Evaluation
# =============================================================================

def evaluate_parity(
    store,
    dates: list[date],
    inst_family: str,
    forwards_recipe,
    binning: Optional[str],
    moneyness_spread: float = 0.05,
    batch_days: Optional[int] = None,
    verbose: bool = True,
) -> pl.LazyFrame:
    """
    Evaluate put-call parity: compare fitted forwards to options-implied forwards.
    
    Put-call parity: C - P = F - K  =>  F_implied = K + (C - P)
    """
    
    # Load options with forwards already assigned
    lf_options = prepare_options(
        store=store,
        inst_family=inst_family,
        dates=dates,
        forwards_recipe=forwards_recipe,
        binning=binning,
        paired=True,
        batch_days=batch_days,
        verbose=verbose
    )
    
    # Filter by moneyness and remove invalid entries
    lf_options = lf_options.filter(
        # Moneyness filter: keep options within ±spread from ATM (moneyness=0)
        (pl.col('moneyness').abs() <= moneyness_spread) &
        # Remove invalid/null entries
        pl.col('call_bid_1_px').is_not_null() &
        pl.col('call_ask_1_px').is_not_null() &
        pl.col('put_bid_1_px').is_not_null() &
        pl.col('put_ask_1_px').is_not_null() &
        (pl.col('call_bid_1_px') > 0) &
        (pl.col('put_bid_1_px') > 0) &
        pl.col('F_bid').is_not_null() &
        pl.col('F_ask').is_not_null()
    )
    
    # Compute parity-implied forwards and errors
    lf_parity = lf_options.with_columns([
        # Implied forwards from parity
        (pl.col('strike') + pl.col('call_bid_1_px') - pl.col('put_ask_1_px')).alias('F_implied_bid'),
        (pl.col('strike') + pl.col('call_ask_1_px') - pl.col('put_bid_1_px')).alias('F_implied_ask'),
        # Mids
        ((pl.col('F_bid') + pl.col('F_ask')) / 2).alias('F_mid'),
    ]).with_columns([
        ((pl.col('F_implied_bid') + pl.col('F_implied_ask')) / 2).alias('F_implied_mid'),
    ]).with_columns([
        # Errors in basis points
        ((pl.col('F_implied_bid') / pl.col('F_bid')).log() * 10000).alias('error_bid_bps'),
        ((pl.col('F_implied_ask') / pl.col('F_ask')).log() * 10000).alias('error_ask_bps'),
        ((pl.col('F_implied_mid') / pl.col('F_mid')).log() * 10000).alias('error_mid_bps'),
        # Option spreads
        (((pl.col('call_ask_1_px') - pl.col('call_bid_1_px')) / 
          ((pl.col('call_ask_1_px') + pl.col('call_bid_1_px')) / 2)) * 10000).alias('call_spread_bps'),
        (((pl.col('put_ask_1_px') - pl.col('put_bid_1_px')) / 
          ((pl.col('put_ask_1_px') + pl.col('put_bid_1_px')) / 2)) * 10000).alias('put_spread_bps'),
    ])
    
    return lf_parity

def summarize_parity(lf_parity: pl.LazyFrame) -> pl.LazyFrame:
    """Compute summary statistics for parity evaluation results."""
    return lf_parity.select([
        pl.count().alias('n_pairs'),
        pl.col('error_mid_bps').mean().alias('error_mid_bps_mean'),
        pl.col('error_mid_bps').std().alias('error_mid_bps_std'),
        pl.col('error_bid_bps').mean().alias('error_bid_bps_mean'),
        pl.col('error_ask_bps').mean().alias('error_ask_bps_mean'),
        pl.col('call_spread_bps').mean().alias('call_spread_bps_mean'),
        pl.col('put_spread_bps').mean().alias('put_spread_bps_mean'),
    ])

# =============================================================================
# Pillar Fit and LOEO Evaluation Helper Functions
# =============================================================================

def _prepare_pillars_for_evaluation(min_time_to_expiry_hours, unique_times, **shared_params) -> pl.LazyFrame:
    return prepare_pillars(
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        **shared_params,
        log_prices=False, # We don't need log prices for this evaluation
    ).filter(pl.col('T') > 0.0).rename({
        'bid_1_px': 'F_bid_obs',
        'ask_1_px': 'F_ask_obs',
    })

def _assign_forwards_and_mids(lf_data, recipe, **shared_params) -> pl.LazyFrame:
    return assign_forwards(
        lf_data=lf_data,
        forwards_recipe=recipe,
        **shared_params
    ).with_columns([
        ((pl.col('F_bid_obs') + pl.col('F_ask_obs')) / 2).alias('F_mid_obs'),
        ((pl.col('F_bid') + pl.col('F_ask')) / 2).alias('F_mid_pred'),
    ]).rename({
        'F_bid': 'F_bid_pred',
        'F_ask': 'F_ask_pred',
    })

def _assign_errors(lf_fitted: pl.LazyFrame) -> pl.LazyFrame:
    return lf_fitted.with_columns([
        ((pl.col('F_bid_pred') / pl.col('F_bid_obs')).log() * 10000).abs().alias('error_bid_bps'),
        ((pl.col('F_ask_pred') / pl.col('F_ask_obs')).log() * 10000).abs().alias('error_ask_bps'),
        ((pl.col('F_mid_pred') / pl.col('F_mid_obs')).log() * 10000).abs().alias('error_mid_bps'),
        # Option spreads
        (((pl.col('F_ask_obs') - pl.col('F_bid_obs')) / 
          ((pl.col('F_ask_obs') + pl.col('F_bid_obs')) / 2)) * 10000).alias('spread_obs_bps'),
        (((pl.col('F_ask_pred') - pl.col('F_bid_pred')) / 
          ((pl.col('F_ask_pred') + pl.col('F_bid_pred')) / 2)) * 10000).alias('spread_pred_bps'),
    ])

# =============================================================================
# Pillar Fit and LOEO Evaluation Functions
# =============================================================================
def evaluate_pillar_fit(
    store,
    dates: list[date],
    inst_family: str,
    forwards_recipe,
    binning: Optional[str],
    min_time_to_expiry_hours: float = 2.0,
    unique_times: Optional[list[int]] = None,
    batch_days: Optional[int] = None,
    verbose: bool = True
) -> pl.LazyFrame:
    """Evaluate pillar fit quality: WMAE between fitted curve and pillar observations."""
    if verbose:
        print(f"Preparing pillars for {len(dates)} dates...")
    shared_params = {
        'store': store,
        'dates': dates,
        'inst_family': inst_family,
        'binning': binning,
        'verbose': verbose,
        'batch_days': batch_days
    }
    # Prepare pillars
    lf_pillars = _prepare_pillars_for_evaluation(
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        **shared_params,
    )
    # Assign forwards to pillars
    lf_fitted = _assign_forwards_and_mids(
        lf_data=lf_pillars,
        recipe=forwards_recipe,
        **shared_params
    )
    return _assign_errors(lf_fitted)

def evaluate_loeo(
    store,
    dates: list[date],
    inst_family: str,
    forwards_recipe,
    binning: Optional[str],
    min_time_to_expiry_hours: float = 2.0,
    unique_times: Optional[list[int]] = None,
    batch_days: Optional[int] = None,
    verbose: bool = True,
) -> pl.LazyFrame:
    """Leave-one-expiry-out cross-validation."""
    if verbose:
        print(f"Running LOEO evaluation for {len(dates)} dates...")
    shared_params = {
        'store': store,
        'dates': dates,
        'inst_family': inst_family,
        'binning': binning,
        'verbose': verbose,
        'batch_days': batch_days
    }
    # Prepare pillars
    lf_pillars = _prepare_pillars_for_evaluation(
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        **shared_params,
    )
    # Partition by pillar_idx
    partitions = lf_pillars.collect().partition_by('pillar_idx', as_dict=True)
    if verbose:
        print(f"Evaluating {len(partitions)} pillars...")
    pillar_iter = tqdm(partitions.items(), desc="LOEO pillar indices", disable=not verbose) if verbose else partitions.items()

    all_results = []
    for (pillar_idx,), df_held_out in pillar_iter:
        # Build recipe with this pillar dropped
        recipe_loeo = partial(forwards_recipe, drop_pillar_idx=pillar_idx)
        # Assign forwards ONCE for all timestamps with this pillar excluded
        lf_result = _assign_forwards_and_mids(
            lf_data=df_held_out.lazy(),
            recipe=recipe_loeo,
            **shared_params
        )
        # Join predictions with observations and compute errors
        all_results.append(lf_result)
    lf_fitted = pl.concat(all_results)
    return _assign_errors(lf_fitted)

def summarize_pillar_fit(lf_errors: pl.LazyFrame) -> pl.LazyFrame:
    """Compute summary statistics for pillar fit and LOEO evaluation results."""
    return lf_errors.select([
        pl.col('error_mid_bps').mean().alias('error_mid_bps_mean'),
        pl.col('error_mid_bps').std().alias('error_mid_bps_std'),
        pl.col('error_bid_bps').mean().alias('error_bid_bps_mean'),
        pl.col('error_ask_bps').mean().alias('error_ask_bps_mean'),
        pl.col('spread_obs_bps').mean().alias('spread_obs_bps_mean'),
        pl.col('spread_pred_bps').mean().alias('spread_pred_bps_mean'),
    ])

# =============================================================================
# Old Evaluation Functions - Might be worth implementing properly in the future
# =============================================================================

# Just keeping docstrings here since the old functions don't fit in the new framework

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

"""
Compute diagnostic statistics for curve quality.

Args:
    errors: Array of errors in bps
    threshold_bps: Threshold for outlier detection
    
Returns:
    EvaluationMetrics with diagnostic statistics
"""