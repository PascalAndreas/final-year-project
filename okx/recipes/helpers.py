"""
Helper functions for recipe pipelines.

Provides reusable filter and transformation functions for building
orderbook processing recipes.
"""

import polars as pl
from typing import Callable, Optional
from functools import partial


def early_roll(min_time_to_expiry_hours: float):
    """Filter out rows too close to expiry (after 'tenor')."""
    def filter_fn(lf: pl.LazyFrame) -> pl.LazyFrame:
        min_seconds = min_time_to_expiry_hours * 3600
        time_to_expiry_seconds = (pl.col('expiry') - pl.col('timeMs')) / 1000.0
        return lf.filter(time_to_expiry_seconds >= min_seconds)
    return filter_fn

def pair_options(lf: pl.LazyFrame, include_qty: bool = False) -> pl.LazyFrame:
    """Pair calls and puts with matching timeMs, expiry, and strike."""
    # Build column lists based on whether quantities are included
    call_cols = ['timeMs', 'expiry', 'strike', 'T', 'bid_1_px', 'ask_1_px']
    put_cols = ['timeMs', 'expiry', 'strike', 'bid_1_px', 'ask_1_px']
    call_rename = {
        'bid_1_px': 'call_bid',
        'ask_1_px': 'call_ask',
    }
    put_rename = {
        'bid_1_px': 'put_bid',
        'ask_1_px': 'put_ask',
    }
    
    if include_qty:
        call_cols.extend(['bid_1_qty', 'ask_1_qty'])
        put_cols.extend(['bid_1_qty', 'ask_1_qty'])
        call_rename.update({
            'bid_1_qty': 'call_bid_qty',
            'ask_1_qty': 'call_ask_qty',
        })
        put_rename.update({
            'bid_1_qty': 'put_bid_qty',
            'ask_1_qty': 'put_ask_qty',
        })
    
    lf_calls = lf.filter(pl.col('opt_type') == 'C').select(call_cols).rename(call_rename)
    lf_puts = lf.filter(pl.col('opt_type') == 'P').select(put_cols).rename(put_rename)
    return lf_calls.join(lf_puts, on=['timeMs', 'expiry', 'strike'], how='inner')

def _format_cache_value(value) -> str:
    """Format parameter value for cache name (e.g., 5.0 -> '5.0', '5m' -> '5m')."""
    if isinstance(value, float):
        return f"{value:.10g}" # Format floats to remove trailing zeros
    elif isinstance(value, (int, str)):
        return str(value)
    elif value is None:
        return "None"
    else:
        return str(value).replace(" ", "") # For other types, use simple string representation

def build_cache_name(binning: str, recipe_type: str, recipe: Callable) -> str:
    """Build cache name like '{binning|full}_{recipe_type}_{params}' for recipes."""
    # Recursively extract all parameters from nested partials
    params = {}
    while isinstance(recipe, partial):
        for key, value in (recipe.keywords or {}).items():
            if key not in params:  # Earlier levels take precedence
                params[key] = value
        recipe = recipe.func
    if params:
        param_parts = [
            f"{key}={_format_cache_value(params[key])}"
            for key in sorted(params)
        ]
        params_str = "_" + "_".join(param_parts)
    else:
        params_str = ""
    return f"{binning if binning is not None else 'full'}_{recipe_type}{params_str}"