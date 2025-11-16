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



def evaluate_parity(
    store,
    dates: list[date],
    inst_family: str,
    forwards_recipe,
    binning: Optional[str],
    moneyness_spread: float = 0.05,
    verbose: bool = True,
) -> pl.LazyFrame:
    """
    Evaluate put-call parity: compare fitted forwards to options-implied forwards.
    
    Put-call parity: C - P = F - K  =>  F_implied = K + (C - P)
    
    Args:
        moneyness_spread: Filter to options with |moneyness| <= spread (default: 0.05).
                         Since moneyness = log(strike/forward), 0 is ATM.
                         Example: 0.05 ≈ ±5% from ATM.
    
    Returns:
        LazyFrame with parity-implied forwards, fitted forwards, and errors
    """
    
    # Load options with forwards already assigned
    lf_options = prepare_options(
        store=store,
        inst_family=inst_family,
        dates=dates,
        forwards_recipe=forwards_recipe,
        binning=binning,
        paired=True,
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

def summarize_parity(lf_parity: pl.LazyFrame) -> pl.DataFrame:
    """
    Compute summary statistics for parity evaluation results.
    
    Args:
        lf_parity: LazyFrame from evaluate_parity()
    
    Returns:
        DataFrame with aggregate statistics
    """
    return lf_parity.select([
        pl.count().alias('n_pairs'),
        pl.col('error_mid_bps').mean().alias('error_mid_bps_mean'),
        pl.col('error_mid_bps').std().alias('error_mid_bps_std'),
        pl.col('error_bid_bps').mean().alias('error_bid_bps_mean'),
        pl.col('error_ask_bps').mean().alias('error_ask_bps_mean'),
        pl.col('call_spread_bps').mean().alias('call_spread_bps_mean'),
        pl.col('put_spread_bps').mean().alias('put_spread_bps_mean'),
    ]).collect()

def evaluate_pillar_fit(
    store,
    dates: list[date],
    inst_family: str,
    forwards_recipe,
    binning: Optional[str],
    min_time_to_expiry_hours: float = 2.0,
    unique_times: Optional[list[int]] = None,
    verbose: bool = True,
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
    }
    # Prepare pillars
    lf_pillars = prepare_pillars(
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        **shared_params
    )
    # Convert back from log prices
    lf_pillars = lf_pillars.with_columns([
        pl.col('ln_bid_1_px').exp().alias('F_bid_obs'),
        pl.col('ln_ask_1_px').exp().alias('F_ask_obs'),
    ]).drop('ln_bid_1_px', 'ln_ask_1_px')
    
    # Assign forwards to pillars
    lf_fitted = assign_forwards(
        lf_data=lf_pillars,
        forwards_recipe=forwards_recipe,
        **shared_params
    ).with_columns([
        ((pl.col('F_bid_obs') + pl.col('F_ask_obs')) / 2).alias('F_mid_obs'),
        ((pl.col('F_bid') + pl.col('F_ask')) / 2).alias('F_mid_pred'),
    ]).rename({
        'F_bid': 'F_bid_pred',
        'F_ask': 'F_ask_pred',
    })
    
    # Compute errors in basis points for each side
    lf_errors = lf_fitted.with_columns([
        ((pl.col('F_bid_pred') / pl.col('F_bid_obs')).log() * 10000).abs().alias('error_bid_bps'),
        ((pl.col('F_ask_pred') / pl.col('F_ask_obs')).log() * 10000).abs().alias('error_ask_bps'),
        ((pl.col('F_mid_pred') / pl.col('F_mid_obs')).log() * 10000).abs().alias('error_mid_bps'),
    ])
    
    return lf_errors

def evaluate_loeo(
    store,
    dates: list[date],
    inst_family: str,
    forwards_recipe,
    binning: Optional[str],
    min_time_to_expiry_hours: float = 2.0,
    unique_times: Optional[list[int]] = None,
    verbose: bool = True,
) -> pl.LazyFrame:
    """
    Leave-one-expiry-out cross-validation.
    
    For each pillar index:
    1. Prepare all pillars with that pillar excluded
    2. Assign forwards once for all timestamps
    3. Compare predictions to observed values
    
    This is much more efficient than the naive approach: instead of calling
    assign_forwards() for every (timeMs, pillar_idx) combination, we call it
    once per pillar_idx.
    
    Returns:
        DataFrame with per-pillar LOEO errors in basis points
    """
    if verbose:
        print(f"Running LOEO evaluation for {len(dates)} dates...")
    
    # Get reference pillars to determine structure (without dropping any)
    lf_pillars_ref = prepare_pillars(
        store=store,
        inst_family=inst_family,
        dates=dates,
        binning=binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        unique_times=unique_times,
        verbose=False,
    )
    
    df_pillars_ref = lf_pillars_ref.collect()
    
    if df_pillars_ref.is_empty():
        return pl.DataFrame()
    
    # Determine max number of pillars to know how many to iterate over
    max_pillars = df_pillars_ref.select("pillar_idx").max().item()
    
    if max_pillars <= 2:
        print("Not enough pillars for LOEO (need at least 3)")
        return pl.DataFrame()
    
    all_results = []
    
    # For each pillar index (skip idx=0, the SWAP anchor)
    pillar_range = range(1, max_pillars)
    pillar_iter = tqdm(pillar_range, desc="LOEO pillar indices", disable=not verbose) if verbose else pillar_range
    
    for pillar_idx in pillar_iter:
        # Get the held-out observations for this pillar_idx
        df_held_out = df_pillars_ref.filter(pl.col('pillar_idx') == pillar_idx).select([
            'timeMs', 'symbol', 'T', 'ln_bid_1_px', 'ln_ask_1_px'
        ])
        
        if df_held_out.is_empty():
            continue
        
        # Build recipe with this pillar dropped (disable EWMA for single-snapshot evaluation)
        recipe_loeo = partial(
            forwards_recipe,
            drop_pillar_idx=pillar_idx,
            tau_ewma_minutes=0.001,  # Disable smoothing
        )
        
        # Assign forwards ONCE for all timestamps with this pillar excluded
        df_pred = assign_forwards(
            store=store,
            lf_data=df_held_out.select(['timeMs', 'T']).lazy(),
            dates=dates,
            forwards_recipe=recipe_loeo,
            inst_family=inst_family,
            binning=binning,
            verbose=False,
        ).collect()
        
        if df_pred.is_empty():
            continue
        
        # Join predictions with observations
        df_joined = df_held_out.join(
            df_pred.select(['timeMs', 'T', 'F_bid', 'F_ask']),
            on=['timeMs', 'T'],
            how='inner'
        )
        
        # Compute errors
        results = df_joined.with_columns([
            pl.col('ln_bid_1_px').exp().alias('F_bid_obs'),
            pl.col('ln_ask_1_px').exp().alias('F_ask_obs'),
        ]).with_columns([
            (((pl.col('F_bid') / pl.col('F_bid_obs')).log()).abs() * 10000).alias('error_bid_bps'),
            (((pl.col('F_ask') / pl.col('F_ask_obs')).log()).abs() * 10000).alias('error_ask_bps'),
        ]).with_columns([
            ((pl.col('error_bid_bps') + pl.col('error_ask_bps')) / 2).alias('error_mid_bps'),
            pl.lit(pillar_idx).alias('pillar_idx'),
            pl.lit(True).alias('success'),
        ]).select([
            'timeMs', 'pillar_idx', 'symbol', 'T',
            'F_bid_obs', 'F_ask_obs',
            pl.col('F_bid').alias('F_bid_pred'),
            pl.col('F_ask').alias('F_ask_pred'),
            'error_bid_bps', 'error_ask_bps', 'error_mid_bps', 'success'
        ])
        
        all_results.append(results)
    
    if not all_results:
        return pl.DataFrame()
    
    return pl.concat(all_results)
