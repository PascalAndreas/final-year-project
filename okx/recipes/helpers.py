"""
Helper functions for recipe pipelines.

Provides reusable filter and transformation functions for building
orderbook processing recipes.
"""

import polars as pl


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