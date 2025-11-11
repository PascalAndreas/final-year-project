"""
Shared data-loading helpers for forward curve analyses.

These wrappers sit on top of store recipes so that evaluation code (notebooks,
tests, CLI tools) can obtain consistent Polars DataFrames without duplicating
fetch/prepare logic.
"""

from datetime import date
from typing import Callable, Optional

import polars as pl

from okx.recipes.options import prepare_options


def load_matched_options(
    store,
    dates: list[date],
    inst_family: str,
    forwards_recipe: Callable,
    binning: Optional[str] = None,
    min_time_to_expiry_hours: float = 2.0,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Return options snapshots with fitted forwards (bid/ask) attached.

    This is a thin wrapper around okx.recipes.options.prepare_options so that
    evaluation pipelines can call a single function without worrying about
    which recipe module to import.
    """
    if not dates:
        return pl.DataFrame()

    return prepare_options(
        store=store,
        inst_family=inst_family,
        dates=dates,
        forwards_recipe=forwards_recipe,
        binning=binning,
        min_time_to_expiry_hours=min_time_to_expiry_hours,
        verbose=verbose,
    )
