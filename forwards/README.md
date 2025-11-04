# Forward Curve Modeling

Exchange-agnostic forward curve modeling infrastructure with evaluation metrics.

## Overview

This module provides two sophisticated methods for building forward curves from futures orderbook data:

1. **PCHIP with EWMA** - Simple, interpretable baseline using piecewise cubic Hermite interpolation
2. **Kalman NS-Carry** - State-space model using Nelson-Siegel factors with Kalman filtering

## Module Structure

```
forwards/
├── __init__.py           # Public API exports
├── utils.py              # Time conversions, weights, early-roll filtering
├── pchip.py              # PCHIP interpolation with EWMA smoothing
├── kalman_ns.py          # Kalman-filtered Nelson-Siegel carry model
├── evaluation.py         # Evaluation metrics (WMAE, LOEO, smoothness, etc.)
└── README.md            # This file
```

## Quick Start

```python
from functools import partial
from okx.store import OrderbookStore
from okx.recipes.forwards import build_forwards_pchip, build_forwards_kalman

# Initialize store
store = OrderbookStore(
    data_root="data/okx",
    manifest_path="data/okx/manifest.sqlite"
)

# Configure PCHIP recipe
pchip_recipe = partial(
    build_forwards_pchip,
    inst_family='BTC-USD',
    binning='5m',
    lambda_ewma=0.8,
    w0_anchor=10.0,
)

# Build forward curve
lf = store.get_derived(
    pchip_recipe,
    cache_name='forwards_pchip_5m',
    start=datetime(2025, 8, 15),
    end=datetime(2025, 8, 16)
)
```

## Methods Comparison

### PCHIP Method

**Advantages:**
- Simple and interpretable
- Fast reconstruction via scipy
- Stable for sparse pillars
- No hyperparameters to tune

**Output:** Node values (T, ln_F_bid, ln_F_ask) at each pillar

**Best for:** Quick analysis, interpretability, when simplicity matters

### Kalman NS Method

**Advantages:**
- Optimal noise filtering
- Compact representation (3 parameters)
- Smooth temporal evolution
- Theoretically grounded

**Output:** State parameters (β₀, β₁, β₂) + λ

**Best for:** Production systems, when smoothness matters, compact storage

## Configuration Parameters

### PCHIP

- `binning`: Time bin frequency ('1m', '5m', None)
- `lambda_ewma`: EWMA smoothing (0.7-0.9, higher = smoother)
- `w0_anchor`: Swap anchor weight (10.0 typical)
- `min_time_to_expiry_hours`: Early-roll threshold (2.0 hours)

### Kalman NS

- `binning`: Time bin frequency ('1m', '5m', None)
- `lambda_ns`: Shape parameter (0.05-0.2, controls curve shape)
- `process_noise_scale`: Process noise (1e-4 typical)
- `ar1_coef`: State persistence (0.99 typical)
- `min_time_to_expiry_hours`: Early-roll threshold (2.0 hours)

## Evaluation Metrics

```python
from forwards.evaluation import (
    wmae_pillar_fit,
    leave_one_expiry_out,
    temporal_smoothness,
    calendar_spread_check,
)

# Pillar fit quality
wmae = wmae_pillar_fit(T_obs, F_obs, F_pred, weights)
print(f"WMAE: {wmae.value:.2f} bps")

# Cross-validation
loeo_df = leave_one_expiry_out(T, F_bid, F_ask, symbols, fit_func, reconstruct_func)

# Temporal smoothness
smoothness_df = temporal_smoothness(forward_series, perp_series, test_maturities, reconstruct_func)

# Calendar spreads
cal_df = calendar_spread_check(T_pillars, F_pred)
```

## Design Principles

1. **Exchange-agnostic**: Core logic in `/forwards/`, exchange-specific orchestration in `/okx/recipes/`
2. **Parameter storage**: Store curve parameters (not dense grids) for efficient reconstruction
3. **Configurable recipes**: Use `functools.partial` to configure recipes before passing to `store.get_derived()`
4. **Comprehensive evaluation**: Multiple metrics to assess quality from different angles

## Example Notebook

See `forwards_example.ipynb` for a complete walkthrough including:
- Building curves with both methods
- Visualization and comparison
- Evaluation and quality metrics
- Temporal analysis

## Dependencies

- `scipy` - PCHIP interpolation
- `polars` - Data manipulation
- `numpy` - Numerical operations
- `okx.helpers` - Symbol parsing utilities

## Future Extensions

- [ ] Add support for other exchanges
- [ ] Implement calendar spread optimization
- [ ] Add more sophisticated Kalman variants (time-varying λ, exogenous inputs)
- [ ] Build forward-to-IV surface pipeline
- [ ] Add real-time evaluation dashboard

