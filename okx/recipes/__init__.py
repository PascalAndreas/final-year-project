"""
Recipes for derived data products using OrderbookStore.

Recipes are functions with signature: (store, start, end, **kwargs) -> LazyFrame
They can be used with store.get_derived() to compute and cache complex derivatives.
"""
from .pillars import prepare_pillars
from .forwards import (
    build_forwards_pchip,
    build_forwards_kalman,
    assign_forwards
)
from .options import (
    prepare_options
)

__all__ = [
    "build_forwards_pchip",
    "build_forwards_kalman",
    "assign_forwards",
    "prepare_pillars",
    "prepare_options",
]
