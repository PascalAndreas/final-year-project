# Forward Pricing API Migration Guide

## Summary of Changes

The forward pricing model has been updated to:
1. **Use bid/ask bounds** instead of midpoints for more accurate modeling
2. **Use perpetual swap** as the anchor instead of spot price
3. **Fully integrate** forward interpolation into IV calculations for ALL options

## API Changes in `forward.py`

### Old API (Midpoint-based)
```python
# Old: Used spot mid price as anchor
segments = compute_segment_carry(futures_df, spot_mid, spot_time)
forward_price = interpolate_forward(T_target, segments)  # Returns single float
curve = compute_continuous_forward_curve(futures_df, spot_mid, spot_time, T_grid)
```

### New API (Bounds-based)
```python
# New: Uses swap bid/ask as anchor, returns bounds
segments = compute_segment_carry(futures_df, swap_bid, swap_ask, swap_time)
F_lower, F_upper = interpolate_forward(T_target, segments)  # Returns (lower, upper) tuple
curve = compute_continuous_forward_curve(futures_df, swap_bid, swap_ask, swap_time, T_grid)

# Helper function to compute mid prices from bounds
curve_with_mid = add_mid_prices(curve)  # Adds F_mid, carry_mid columns
```

## Data Fetching Changes

### Old: Fetch with depth=0 (mid prices only)
```python
swap_df = fetch_market_data('6', 'SWAP', 'BTC-USD', start, end, 'daily', depth=0, ...)
futures_df = fetch_market_data('6', 'FUTURES', 'BTC-USD', start, end, 'daily', depth=0, ...)
```

### New: Fetch with depth=1 (includes bid/ask)
```python
swap_df = fetch_market_data('6', 'SWAP', 'BTC-USD', start, end, 'daily', depth=1, ...)
futures_df = fetch_market_data('6', 'FUTURES', 'BTC-USD', start, end, 'daily', depth=1, ...)
```

## Example: Updating `futures_pricing.ipynb`

### Cell 8 - Compute segment carry (OLD)
```python
example_time = binned_swap_df['time_bin'].iloc[0]
swap_mid = binned_swap_df[binned_swap_df['time_bin'] == example_time]['mid_price'].iloc[0]
futures_at_time = binned_futures_df[binned_futures_df['time_bin'] == example_time].copy()

segments = compute_segment_carry(futures_at_time, swap_mid, example_time)
```

### Cell 8 - Compute segment carry (NEW)
```python
example_time = binned_swap_df['time_bin'].iloc[0]
swap_row = binned_swap_df[binned_swap_df['time_bin'] == example_time].iloc[0]
swap_bid, swap_ask = swap_row['bid_1_px'], swap_row['ask_1_px']
futures_at_time = binned_futures_df[binned_futures_df['time_bin'] == example_time].copy()

# Optional: drop contracts within 4 hours of expiry (useful for plotting)
segments = compute_segment_carry(futures_at_time, swap_bid, swap_ask, example_time, min_time_to_expiry=4.0)
# To view mid prices alongside bounds:
segments = add_mid_prices(segments)
print(segments[['segment_num', 'symbol_start', 'symbol_end', 'T_start', 'T_end', 
                'carry_lower', 'carry_upper', 'carry_mid']])
```

### Cell 9 - Compute forward curve (OLD)
```python
forward_curve = compute_continuous_forward_curve(futures_at_time, swap_mid, example_time, T_grid)
# Plot
ax.scatter(observed['T'], observed['F'], ...)
ax.plot(interpolated['T'], interpolated['F'], ...)
```

### Cell 9 - Compute forward curve (NEW)
```python
forward_curve = compute_continuous_forward_curve(futures_at_time, swap_bid, swap_ask, example_time, T_grid)
forward_curve = add_mid_prices(forward_curve)

# Plot with bounds
observed = forward_curve[forward_curve['source'] == 'observed']
interpolated = forward_curve[forward_curve['source'].isin(['interpolated', 'swap'])]

# Show bounds as shaded region
ax.fill_between(interpolated['T'], interpolated['F_lower'], interpolated['F_upper'], 
                alpha=0.2, label='Forward Bounds')
ax.plot(interpolated['T'], interpolated['F_mid'], 'b-', linewidth=2, label='Mid Forward')
ax.scatter(observed['T'], observed['F_mid'], color='red', s=100, zorder=3, label='Observed')
```

## Changes in `iv.py`

The IV surface construction now:
- Fetches SWAP data automatically
- Computes forwards for **ALL option expiries** via interpolation (not just those with listed futures)
- No longer requires manual filtering of options by available futures

### Old Behavior
```python
# Only computed IV for options with corresponding futures
iv_df = construct_iv_surface('BTC-USD', start_date, num_days=1, ...)
# Result: ~1000 IV points (limited by futures availability)
```

### New Behavior
```python
# Computes IV for ALL options using interpolated forwards
iv_df = construct_iv_surface('BTC-USD', start_date, num_days=1, ...)
# Result: ~5000 IV points (all option expiries)
```

## Column Name Changes

### Segment DataFrame
- **Old**: `F_start`, `F_end`, `carry_rate`, `carry_rate_pct`, `symbol_start='SPOT'`
- **New**: `F_lower_start`, `F_lower_end`, `F_upper_start`, `F_upper_end`, 
  `carry_lower`, `carry_upper`, `carry_lower_pct`, `carry_upper_pct`, `symbol_start='SWAP'`

### Forward Curve DataFrame
- **Old**: `F` (single column)
- **New**: `F_lower`, `F_upper` (use `add_mid_prices()` to get `F_mid`)

## Key Benefits

1. **More accurate pricing**: Bid-ask bounds capture market microstructure
2. **Better anchor**: Perpetual swap is actively traded 24/7 (more liquid than spot)
3. **Complete IV surface**: No missing expiries due to lack of listed futures
4. **Consistent modeling**: Same forward interpolation used throughout IV calculations

