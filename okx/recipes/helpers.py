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

def finalize_binning(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Finalize binning by promoting time_bin to timeMs and dropping old timeMs column."""
    return (lf
        .drop('timeMs')
        .with_columns(pl.col('time_bin').dt.epoch('ms').alias('timeMs'))
        .drop('time_bin')
    )

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