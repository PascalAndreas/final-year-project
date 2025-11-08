"""
Recipes for derived data products using OrderbookStore.

Recipes are functions with signature: (store, start, end, **kwargs) -> LazyFrame
They can be used with store.get_derived() to compute and cache complex derivatives.
"""

from .forwards import (
    build_forwards_pchip,
    build_forwards_kalman,
    prepare_pillars,
)
from .options import (
    build_forwards_options_comparison,
    prepare_options
)

__all__ = [
    "build_forwards_pchip",
    "build_forwards_kalman",
    "prepare_pillars",
    "build_forwards_options_comparison",
    "prepare_options",
]
