# Forwards Package

Production-ready forward curve fitting for crypto derivatives using orderbook data.

## Overview

This package provides two complementary methods for building forward curves from futures orderbook data:

1. **PCHIP + EWMA**: Exact pillar fitting with temporal smoothing
2. **Kalman-NS**: Smooth curves using Nelson-Siegel carry model with Kalman filtering

Both implementations are **time-aware** and **frame-rate invariant**, producing consistent results regardless of data binning interval (1m, 5m, irregular timestamps).

## Core Components

### Data Classes

#### `PCHIPCurve`
Represents a forward curve fitted using PCHIP interpolation.

**Attributes**:
- `timeMs`: Timestamp in milliseconds
- `T_nodes`: Time-to-maturity nodes (years)
- `ln_F_bid_nodes`, `ln_F_ask_nodes`: Log-forward values at nodes
- `symbols`: Instrument symbols for each node
- `source`: Source type ('swap', 'observed')

**API**:
```python
# Create from DataFrame
curve = PCHIPCurve.from_polars(df)  # Single curve or list
curves = PCHIPCurve.from_polars(df_all)  # Multiple curves

# Convert to DataFrame
df = curve.to_polars()

# Reconstruct forwards
F_bid, F_ask = reconstruct_forward(curve, T_target=0.25)  # 3-month forward
```

#### `NSCarryState`
Represents Nelson-Siegel carry state from Kalman filter.

**Attributes**:
- `timeMs`: Timestamp in milliseconds
- `beta0`, `beta1`, `beta2`: Nelson-Siegel factors
- `lambda_ns`: Shape parameter
- `F_ref_bid`, `F_ref_ask`: Reference forwards

**API**:
```python
# Create from DataFrame
state = NSCarryState.from_polars(df)  # Single state or list
states = NSCarryState.from_polars(df_all)  # Multiple states

# Convert to DataFrame
df = state.to_polars()

# Reconstruct forwards
F_bid, F_ask = reconstruct_ns_forward(state, T_target=0.25)
```

### Fitting Functions

#### PCHIP Interpolation

```python
from forwards import fit_pchip_curve, EWMAState

# Fit single curve
curve = fit_pchip_curve(
    T_pillars=np.array([0.08, 0.25, 0.5, 1.0]),
    F_bid_pillars=bid_prices,
    F_ask_pillars=ask_prices,
    symbols=['BTC-USD-250905.OK', ...],
    timeMs=timestamp,
    T_swap=0.0,
    F_bid_swap=perp_bid,
    F_ask_swap=perp_ask,
    w0_anchor=10.0,  # Soft anchor weight
)

# Apply temporal smoothing
ewma = EWMAState(tau_minutes=5.0)  # 5-minute time constant
for curve in curves:
    smoothed = ewma.update(curve)
```

**Parameters**:
- `tau_minutes`: Time constant for EWMA (default: 5.0)
  - Higher = smoother but more lag
  - Half-life ≈ 0.693 × tau
  - Typical: 3-10 minutes for crypto

#### Kalman-NS Filtering

```python
from forwards import kalman_filter

states = kalman_filter(
    snapshots=snapshots,  # List of dicts with pillar data
    lambda_ns=1.0,  # Shape parameter
    tau_minutes=np.array([2880, 7200, 14400]),  # [2d, 5d, 10d] time constants
    sigma_per_sqrt_day=np.array([0.01, 0.01, 0.01]),  # Factor volatilities
    kappa_spread=0.5,  # Spread-based noise scaling
)
```

**Parameters**:
- `lambda_ns`: Shape parameter (0.5-2.0/year for crypto)
- `tau_minutes`: Mean reversion times for [β0, β1, β2] factors
  - Default: [2880, 7200, 14400] minutes = [2d, 5d, 10d]
- `sigma_per_sqrt_day`: Factor volatilities (innovation size)
  - Default: [0.01, 0.01, 0.01]
- `kappa_spread`: Scales bid-ask spread to measurement noise
  - Default: 0.5 (moderate trust in tight spreads)

### Recipes (High-Level API)

#### Building Curves from Store

```python
from okx.store import OrderbookStore
from okx.recipes.forwards import build_forwards_pchip, build_forwards_kalman
from functools import partial
from datetime import date

store = OrderbookStore('/path/to/data/okx')

# PCHIP recipe
pchip_recipe = partial(
    build_forwards_pchip,
    inst_family='BTC-USD',
    binning='5m',  # Or '1m', '10m', None
    tau_ewma_minutes=5.0,
    min_time_to_expiry_hours=2.0,
)

df_pchip = pchip_recipe(
    store,
    dates=[date(2025, 9, 1), date(2025, 9, 2)]
).collect()

# Kalman recipe
kalman_recipe = partial(
    build_forwards_kalman,
    inst_family='BTC-USD',
    binning='1m',
    lambda_ns=1.0,
    min_time_to_expiry_hours=2.0,
)

df_kalman = kalman_recipe(
    store,
    dates=[date(2025, 9, 1)]
).collect()
```

#### Custom Timestamps

```python
# Build curves at specific times (e.g., for option pricing)
option_times = [1725177600000, 1725181200000, ...]  # Option observation times

df_curves = build_forwards_pchip(
    store,
    dates=[date(2025, 9, 1)],
    binning=None,
    unique_times=option_times,  # Override binning
)
```

### Evaluation

```python
from forwards import evaluate_curve_snapshot

# Evaluate curve quality
metrics = evaluate_curve_snapshot(
    T_pillars=T_obs,
    F_bid_obs=F_bid_obs,
    F_ask_obs=F_ask_obs,
    F_bid_pred=F_bid_pred,
    F_ask_pred=F_ask_pred,
    spreads=spreads,
)

print(f"WMAE: ${metrics['wmae']:.2f} ({metrics['wmae_bps']:.2f} bps)")
```

### Visualization

```python
from forwards import (
    plot_pchip_snapshot,
    plot_kalman_snapshot,
    plot_ns_factors,
    plot_forward_tracking,
    create_comparison_figure,
)

# Single curve snapshots
plot_pchip_snapshot(df_pchip, timeMs=target_time)
plot_kalman_snapshot(state)

# Time series
plot_ns_factors(df_kalman)
plot_forward_tracking(df_kalman, tenors=[0.25, 0.5, 1.0])

# Comprehensive comparison
fig, axes = create_comparison_figure(df_pchip, df_kalman, snapshot_idx=50)
```

## Time-Aware Design

### What is Frame-Rate Invariance?

Both methods use continuous-time parameterization to ensure consistent behavior regardless of data sampling frequency:

**EWMA**: `α(Δt) = exp(-Δt/τ)`
- Same smoothing dynamics at 1m, 5m, or irregular timestamps
- Handles gaps correctly (large Δt → fast realignment)

**Kalman**: Exact Ornstein-Uhlenbeck discretization
```
dβ_k = -(1/τ_k)·β_k·dt + σ_k·dW_k

A_k = exp(-Δt/τ_k)
Q_k = (σ_k²·τ_k/2)·(1 - exp(-2Δt/τ_k))
```

**Benefits**:
- Build at 5m for storage efficiency, 1m for analysis → same results
- Price options at arbitrary times without special handling
- Market gaps handled automatically

### Example

```python
# These produce consistent curves at common timestamps
curve_1m = build_forwards_kalman(store, dates, binning='1m')
curve_5m = build_forwards_kalman(store, dates, binning='5m')

# At t = 10min:
# - 1m filter has processed 10 observations
# - 5m filter has processed 2 observations
# → Estimates differ (expected! 1m has more information)
# BUT: dynamics (A, Q) scale correctly with Δt
```

## Method Comparison

| Feature | PCHIP + EWMA | Kalman-NS |
|---------|-------------|-----------|
| **Pillar fit** | Exact (0 bps) | Small error (~2-5 bps) |
| **Smoothness** | Good with EWMA | Very good (model-based) |
| **Robustness** | Sensitive to outliers | Robust (Kalman smoothing) |
| **Speed** | Very fast | Fast |
| **Interpretability** | High (direct interpolation) | Medium (factor model) |
| **Missing data** | Requires interpolation | Handles naturally |
| **Use case** | Pricing, clean markets | Research, gappy data, forecasting |

## Typical Workflow

```python
from datetime import date
from functools import partial
from okx.store import OrderbookStore
from okx.recipes.forwards import build_forwards_pchip

# 1. Setup
store = OrderbookStore('/path/to/data/okx')
dates = [date(2025, 9, 1)]

# 2. Build curves
recipe = partial(
    build_forwards_pchip,
    binning='5m',
    tau_ewma_minutes=5.0,
)
df_curves = recipe(store, dates).collect()

# 3. Extract curve for specific time
curve = PCHIPCurve.from_polars(
    df_curves.filter(pl.col('timeMs') == target_time)
)

# 4. Price forward at arbitrary maturity
F_bid, F_ask = reconstruct_forward(curve, T_target=0.25)
print(f"3M forward: bid={F_bid:.2f}, ask={F_ask:.2f}")

# 5. Evaluate
from forwards import evaluate_curve_snapshot

metrics = evaluate_curve_snapshot(...)
print(f"WMAE: {metrics['wmae_bps']:.2f} bps")
```

## Module Structure

```
forwards/
├── __init__.py              # Package exports
├── pchip.py                 # PCHIP fitting + EWMA
├── kalman_ns.py             # Kalman-NS filtering
├── evaluation.py            # Metrics and diagnostics
├── plotting.py              # Visualization tools
├── utils.py                 # Shared utilities
└── README.md                # This file
```

## Performance Notes

**PCHIP**:
- Fitting: ~0.1ms per curve
- EWMA update: ~0.05ms per curve
- Recommended for: Real-time pricing, clean markets

**Kalman-NS**:
- Filtering: ~0.5ms per snapshot
- Recommended for: Historical analysis, research, gappy data

**Memory**:
- PCHIP state: ~1KB per curve
- Kalman state: ~200B per snapshot

## Future Enhancements

- [ ] Leave-one-expiry-out (LOEO) evaluation
- [ ] Options-implied forward comparison
- [ ] Calendar spread reconstruction checks
- [ ] Grid search for parameter tuning
- [ ] Soft anchor for Kalman filter
- [ ] RTS smoother for historical analysis (non-causal)

## References

**Nelson-Siegel Model**:
- Nelson & Siegel (1987): "Parsimonious Modeling of Yield Curves"
- Diebold & Li (2006): "Forecasting the term structure"

**Time-Series Methods**:
- Anderson (1982): "Optimal Filtering" (OU discretization)
- Shumway & Stoffer (2017): "Time Series Analysis and Its Applications"

**PCHIP**:
- Fritsch & Carlson (1980): "Monotone Piecewise Cubic Interpolation"
