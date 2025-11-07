# Forward Curve Development Plan

## Current Status: Phase 1 Complete ✓

### Completed Work

**Phase 1: Simplify & Refine `/forwards` Package** ✅

All refinements to the forwards package have been completed:

1. ✅ **Unified PCHIPCurve and NSCarryState API**
   - Both classes now use `from_polars()` and `to_polars()` methods
   - `from_polars()` handles single-row → single object, multi-row → list of objects
   - Consistent API pattern across both curve types

2. ✅ **Moved log-price logic upstream**
   - `prepare_pillars()` now includes `'log'` feature in its pipeline
   - Outputs `ln_bid_1_px`, `ln_ask_1_px` columns
   - Fit functions now expect pre-computed log prices

3. ✅ **Created data conversion helpers**
   - Added `_extract_swap_ref()`, `_extract_pillars()`, `_pillar_snapshot_to_dict()`, `_concat_swap_and_pillars()` in `okx/recipes/forwards.py`
   - These are private helper functions scoped to the recipes module
   - Significantly reduced verbosity in recipe loops

4. ✅ **Simplified `fit_pchip_curve()`**
   - Removed internal swap concatenation logic
   - Removed unused weighting logic
   - Removed redundant `source` field from PCHIPCurve
   - Function now ~40 lines (down from ~90)
   - Concatenation handled upstream in recipes

5. ✅ **Added LOEO hooks**
   - `prepare_pillars()` accepts `drop_pillar_idx` parameter
   - Drops specified pillar after sorting futures by T
   - Enables efficient leave-one-expiry-out evaluation without reloading data

6. ✅ **Additional cleanup**
   - Deleted `forwards/utils.py` (functionality covered by `okx/helpers.py`)
   - Deleted `forwards/data_utils.py` (moved to recipes as private functions)
   - Deleted migration/implementation docs (no longer needed)
   - Removed unused `compute_weights()` function

**Time-Aware Implementation** ✅

Both curve methods now use frame-rate invariant parameters:

- **PCHIP + EWMA**: `α(Δt) = exp(-Δt/τ)` where `τ` is time constant in minutes
- **Kalman-NS**: Exact OU discretization with `tau_minutes` and `sigma_per_sqrt_day`
- Both handle irregular timestamps and gaps correctly

---

## Next Steps

### Phase 1: Evaluations & Options Comparison ✅

**Status**: COMPLETE

Both evaluation functions have been implemented and tested:

1. **LOEO Evaluation** (`loeo_error`) ✅
   - Integrates with `prepare_pillars` and `drop_pillar_idx`
   - Works with any forward curve recipe
   - Tested with PCHIP: ~1.88 bps median error on real data
   - Returns detailed error statistics for each pillar

2. **Options Comparison** (`build_forwards_options_comparison`) ✅
   - Compares fitted forwards to options-implied forwards via put-call parity
   - Uses robust bid-ask handling
   - Filters by moneyness and time-to-expiry
   - Returns comprehensive comparison metrics

#### 1.1 Options Comparison Recipe ✅

**Priority**: High  
**Estimated time**: 2-3 hours (COMPLETED)

**Goal**: Compare fitted forward curves to options-implied forwards via put-call parity

**New file**: `okx/recipes/options_eval.py`

**Implementation**:

```python
def build_forwards_options_comparison(
    store,
    dates: list[date],
    forwards_recipe: Callable,  # e.g., partial(build_forwards_pchip, ...)
    inst_family: str = 'BTC-USD',
    min_moneyness: float = 0.9,
    max_moneyness: float = 1.1,
) -> pl.LazyFrame:
    """
    Compare fitted forwards to options-implied forwards.
    
    1. Load option orderbook for date range
    2. Extract unique timestamps from options
    3. Build forward curves at those timestamps using forwards_recipe
    4. For each option pair (call+put at same strike/expiry):
       - Extract implied forward from put-call parity: F = K + (C - P) * exp(rT)
       - Compare to fitted forward at that expiry: F_fitted(T_expiry)
       - Compute error in bps
    5. Return DataFrame with comparison metrics
    
    Returns DataFrame with columns:
        - timeMs: Observation timestamp
        - expiry_dt: Option expiry
        - strike: Strike price
        - T: Time to maturity (years)
        - F_fitted_bid: Fitted forward bid price
        - F_fitted_ask: Fitted forward ask price
        - F_implied: Options-implied forward (from put-call parity)
        - error_bps: (F_implied - F_fitted_mid) / F_fitted_mid * 10000
        - call_bid_1_px, call_ask_1_px, put_bid_1_px, put_ask_1_px
    """
```

**Key considerations**:
- Filter options by moneyness to avoid illiquid wings
- Handle missing pairs (when call or put not available)
- Weight errors by option liquidity (inverse bid-ask spread)
- Group by expiry to analyze per-pillar vs interpolated regions

#### 1.2 LOEO Evaluation Function ✅

**Priority**: High  
**Estimated time**: 1-2 hours (COMPLETED)

**Goal**: Systematic leave-one-expiry-out cross-validation

**Implemented in `forwards/evaluation.py`**:

```python
def loeo_error(
    store,
    dates: list[date],
    forwards_recipe: Callable,
    recipe_kwargs: dict = None,
) -> pl.DataFrame:
    """
    Compute leave-one-expiry-out error for forward curve.
    
    For each pillar:
        1. Rebuild curve with that pillar excluded (using drop_pillar_idx)
        2. Predict F(T_pillar) from fitted curve
        3. Compare to observed F_pillar
        4. Compute error in bps
    
    Returns DataFrame with columns:
        - timeMs: Observation timestamp
        - pillar_idx: Index of dropped pillar
        - symbol: Contract symbol (e.g., 'BTC-USD-251031.OK')
        - T: Time to maturity (years)
        - F_obs_bid, F_obs_ask: Observed prices
        - F_pred_bid, F_pred_ask: Predicted prices
        - error_bid_bps, error_ask_bps: Errors in basis points
    """
```

**Process**:
1. Get number of pillars for each snapshot
2. Loop over pillar indices (0 to N-1)
3. Rebuild curve with `drop_pillar_idx=i`
4. Reconstruct forward at dropped pillar's T
5. Compute error vs observed

---

### Phase 2: Plotting Improvements

#### 2.1 Interpretable Plot Functions

**Priority**: Medium  
**Estimated time**: 2-3 hours

**Goal**: Replace F(T) plots with interpretable metrics

**New functions in `forwards/plotting.py`**:

```python
def plot_carry_curve(curve, ax=None):
    """
    Plot annualized carry: c(T) = d(ln F)/dT
    
    Shows:
        - Bid carry (green)
        - Ask carry (red)
        - Mid carry (black dashed)
    
    Interpretable: Positive carry = contango, negative = backwardation
    """

def plot_calendar_spreads(curve, observed_pillars, ax=None):
    """
    Plot calendar spreads: ln(F_j+1) - ln(F_j)
    
    Shows:
        - Fitted spreads (line)
        - Observed spreads (scatter with error bars)
    
    Tests if curve reproduces observed spreads within noise
    """

def plot_loeo_errors(loeo_df, ax=None):
    """
    Visualize LOEO cross-validation errors
    
    Shows:
        - Histogram of errors (bps)
        - Time series of errors colored by pillar
        - Summary statistics box
    """

def plot_options_comparison(options_df, ax=None):
    """
    Compare fitted vs implied forwards
    
    Shows:
        - Scatter: F_fitted vs F_implied
        - Error distribution histogram
        - Error by moneyness
        - Error by time to expiry
    """

def plot_temporal_smoothness(curves_df, tenor_years=0.25, ax=None):
    """
    Plot smoothness at fixed tenor
    
    Shows:
        - F(T*) time series
        - ΔF(T*) time series
        - Volatility of changes
    """
```

#### 2.2 Performance Optimization

**Priority**: Low  
**Estimated time**: 1 hour

**Changes**:
- Pre-allocate subplot grids
- Cache PCHIP interpolators where reused
- Reduce redundant F(T) evaluations
- Use vectorized operations for carry/spread calculations

---

### Phase 3: Comprehensive Evaluation Notebook

#### 3.1 New Notebook: `forwards_evaluation.ipynb`

**Priority**: High  
**Estimated time**: 2-3 hours

**Structure**:

```markdown
# Forward Curve Evaluation Suite

## 1. Setup
- Load store
- Define test period (e.g., September 2025)
- Configure PCHIP and Kalman recipes

## 2. Build Curves
- Build PCHIP curves
- Build Kalman curves
- Compare build times

## 3. In-Sample Metrics

### 3.1 Pillar Fit (WMAE)
- WMAE at pillar locations
- Should be ~0 for PCHIP, small for Kalman

### 3.2 Temporal Smoothness
- Variance of Δln F(T) at fixed tenors (1D, 1W, 1M, 3M)
- Lower = smoother (but might lag)

### 3.3 Calendar Spread Reconstruction
- MAE between fitted and observed calendar spreads
- Should be < spread/2 for good fit

## 4. Out-of-Sample Metrics

### 4.1 LOEO Cross-Validation
- Leave-one-expiry-out errors
- PCHIP vs Kalman comparison
- Breakdown by pillar maturity (front vs back)

## 5. Options Comparison

### 5.1 Fitted vs Implied Forwards
- Scatter plot with identity line
- Error distribution (should be tight)

### 5.2 Error by Moneyness
- Does error increase off ATM?

### 5.3 Error by Maturity
- Near pillars vs interpolated regions

## 6. Performance

### 6.1 Latency
- Time to build 1000 snapshots
- PCHIP should be ~10-50x faster

### 6.2 Memory
- Peak memory usage

## 7. Summary & Recommendations

Decision matrix:
- PCHIP + EWMA: For pricing (fast, accurate)
- Kalman: For robust mode, forecasting, research
```

#### 3.2 Light Kalman Grid Search

**Priority**: Medium  
**Estimated time**: 1-2 hours

**Avoid rabbit hole**: Quick 3×3×3 = 27 runs only

**Parameters**:
- `lambda_ns`: [0.5, 1.0, 2.0] (Nelson-Siegel decay)
- `tau_scale`: [0.5, 1.0, 2.0] (multiply default `tau_minutes`)
- `sigma_scale`: [0.5, 1.0, 2.0] (multiply default `sigma_per_sqrt_day`)

**Objective**: 
```
Score = WMAE_loeo + λ * smoothness_penalty
where λ ≈ 0.1 to balance fit vs smoothness
```

**Process**:
1. Run LOEO on one day of data (e.g., 2025-09-15)
2. Compute score for each parameter combo
3. If best score improves by >20% over defaults, update `forwards/kalman_ns.py` defaults
4. Document in notebook

**Output**:
- Heatmap of scores
- Best parameters
- Pareto frontier: smoothness vs LOEO error

---

### Phase 4: IV Surface Recipe (Future)

**Priority**: Low (post-forwards)  
**Estimated time**: 4-6 hours

**New file**: `okx/recipes/iv_surface.py`

**Key idea**: IV surface recipe takes `forwards_recipe` as parameter

```python
def build_iv_surface(
    store,
    dates: list[date],
    forwards_recipe: Callable,  # e.g., partial(build_forwards_pchip, ...)
    inst_family: str = 'BTC-USD',
    surface_model: str = 'svi',  # or 'sabr', 'rbf'
) -> pl.LazyFrame:
    """
    Build IV surface using fitted forward curves for pricing.
    
    1. Load option orderbook
    2. Build forward curves at option timestamps
    3. Price options using Black-76 with fitted forwards
    4. Extract implied volatilities
    5. Fit surface model (SVI, SABR, RBF)
    6. Return surface parameters + diagnostics
    """
```

---

## Success Criteria

### Phase 1 (Evaluations) ✅
- [x] Options comparison recipe working
- [x] LOEO evaluation showing PCHIP vs Kalman performance
- [x] Error distributions documented
- [x] Test notebook created (`phase1_evaluation_test.ipynb`)

### Phase 2 (Plotting)
- [ ] Carry, calendar spread, LOEO, options plots implemented
- [ ] Old F(T) plots de-emphasized or removed
- [ ] Plotting performance acceptable (<1s per figure)

### Phase 3 (Evaluation Notebook)
- [ ] Comprehensive evaluation notebook complete
- [ ] All metrics computed for test period
- [ ] Kalman grid search run (if pursuing)
- [ ] Summary table with recommendations
- [ ] Decision: "Ship PCHIP + EWMA for pricing"

### Phase 4 (IV Surface)
- [ ] IV surface recipe accepts forwards_recipe parameter
- [ ] SVI or SABR model implemented
- [ ] Surface diagnostics (arbitrage checks, smoothness)
- [ ] Ready for production use

---

## Implementation Notes

### Testing Strategy

For each new component:
1. Unit test on synthetic data first
2. Smoke test on real data (1 day)
3. Full evaluation on test period (1 month)

### Code Quality

- Keep functions focused and modular
- Use type hints for all function signatures
- Document edge cases and assumptions
- Profile before optimizing (don't guess)

### Decision Points

**After Phase 1**:
- If PCHIP LOEO is very good (< 5 bps median error), proceed with PCHIP + EWMA as primary
- If Kalman shows significant LOEO improvement, consider hybrid approach

**After Phase 3**:
- If options comparison shows systematic bias, revisit anchor/weighting
- If performance is unacceptable, parallelize curve building

**Before Phase 4**:
- Validate that forward curves are production-ready
- Ensure evaluation shows stable, predictable behavior
- Get sign-off that curve quality is sufficient for IV extraction

---

## Timeline Estimate

| Phase | Component | Hours |
|-------|-----------|-------|
| 1.1 | Options comparison recipe | 2-3 |
| 1.2 | LOEO evaluation function | 1-2 |
| 2.1 | Interpretable plots | 2-3 |
| 2.2 | Plot optimization | 1 |
| 3.1 | Evaluation notebook | 2-3 |
| 3.2 | Kalman grid search | 1-2 |
| **Total (Phases 1-3)** | | **9-14 hours** |
| 4 | IV surface recipe | 4-6 |
| **Grand Total** | | **13-20 hours** |

---

## Current State of Codebase

### Package Structure

```
forwards/
├── __init__.py          # Public API exports
├── pchip.py             # PCHIP curve fitting (clean, ~130 lines)
├── kalman_ns.py         # Kalman-NS implementation (~450 lines)
├── evaluation.py        # Metrics functions (~400 lines)
├── plotting.py          # Visualization (~330 lines)
└── README.md            # Package documentation

okx/
├── store.py             # Data storage and caching
├── helpers.py           # Parsing, transformations
├── api.py               # OKX API client
└── recipes/
    └── forwards.py      # Forward curve recipes (~400 lines)
```

### Key Dependencies

- `polars` for data manipulation
- `numpy` for numerical computation
- `scipy` for PCHIP interpolation
- `matplotlib` for plotting
- OKX orderbook data via `OrderbookStore`

### Data Flow

```
OrderbookStore (raw parquet files)
    ↓
prepare_pillars() [swap + futures with features]
    ↓
build_forwards_pchip() or build_forwards_kalman()
    ↓
PCHIPCurve or NSCarryState objects
    ↓
to_polars() → LazyFrame for analysis/storage
```

---

## Contact & Questions

For questions or clarifications on this plan, refer to:
- `forwards/README.md` for package usage
- `okx/STORAGE_README.md` for data storage details
- Git history for implementation details of Phase 1 work

