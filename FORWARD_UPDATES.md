# Forward Pricing Updates

## Summary of Changes

### 1. **Added Segment Numbering**
All segments now have a `segment_num` field (1, 2, 3, ...) that allows tracking continuation when contracts roll.

### 2. **Added Near-Expiry Filtering**
New optional parameter `min_time_to_expiry` (in hours) filters out contracts close to expiry to avoid carry rate spikes as T→0.

### 3. **Removed _pct Columns**
Removed redundant `carry_lower_pct`, `carry_upper_pct`, `carry_mid_pct` columns from the output. Use `carry_lower * 100`, etc. for percentages.

## Updated Function Signatures

```python
def compute_segment_carry(
    futures_df: pd.DataFrame, 
    swap_bid: float, 
    swap_ask: float, 
    swap_time: pd.Timestamp,
    min_time_to_expiry: Optional[float] = None  # NEW: in hours (e.g., 4.0)
) -> pd.DataFrame
```

**New columns in output**:
- `segment_num`: Integer (1, 2, 3, ...) identifying the segment

**Example**:
```python
# Drop contracts within 4 hours of expiry
segments = compute_segment_carry(futures_df, swap_bid, swap_ask, swap_time, min_time_to_expiry=4.0)
```

## Usage in Plotting

### Ordinal Segment Labels

Instead of using contract-specific labels (e.g., "SWAP → BTC-USD-240927"), use ordinal labels:

```python
# Create ordinal labels (1st, 2nd, 3rd, etc.)
def ordinal(n):
    return f"{n}{'th' if 11<=n<=13 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

all_segments_df['segment_label'] = all_segments_df['segment_num'].apply(ordinal)

# Plot by segment number (shows continuation across contract rolls)
for segment_num in sorted(all_segments_df['segment_num'].unique()):
    segment_data = all_segments_df[all_segments_df['segment_num'] == segment_num]
    segment_label = segment_data['segment_label'].iloc[0]
    
    plt.plot(segment_data['time_bin'], segment_data['carry_mid'] * 100, 
             label=segment_label)
```

This creates plots with labels like "1st", "2nd", "3rd", "4th", "5th", "6th" instead of hundreds of contract-specific labels.

## Why These Changes?

### 1. Segment Numbering
**Problem**: When futures contracts expire and roll, the old approach treated "SWAP → BTC-USD-240927" and "SWAP → BTC-USD-241025" as completely different segments, even though they represent the same economic relationship (SWAP to front-month).

**Solution**: Ordinal numbering (1st segment = SWAP to nearest futures, 2nd segment = nearest to second-nearest, etc.) shows the continuation clearly.

### 2. Near-Expiry Filtering
**Problem**: As T→0, carry rates explode due to small denominator: `carry = ln(F_end/F_start) / (T_end - T_start)`. This creates massive spikes in plots (e.g., 30,000% carry rates).

**Solution**: Filter out contracts within 4 hours of expiry. These short-dated forwards aren't inaccurate, but the annualized carry rate becomes meaningless for visualization.

### 3. Why 4 Hours?
- Most futures expire at 08:00 UTC
- 4 hours = ~1/2190 of a year
- Small price differences create wild carry rates when denominator is this small
- Not needed for IV calculations (interpolation works fine), but essential for clean plots

## Example: Updated forward_candles.ipynb

```python
# Cell 4: Single timestamp
segments = compute_segment_carry(
    futures_at_time, swap_bid, swap_ask, example_time, 
    min_time_to_expiry=4.0  # Drop contracts < 4h to expiry
)
segments = add_mid_prices(segments)

print(segments[['segment_num', 'symbol_start', 'symbol_end', 
                'T_start', 'T_end', 'carry_lower', 'carry_upper', 'carry_mid']])

# Cell 6: Full time series
for _, swap_row in tqdm(swap_df.iterrows()):
    time_bin = swap_row['time_bin']
    swap_bid, swap_ask = swap_row['bid_1_px'], swap_row['ask_1_px']
    futures_at_time = futures_df[futures_df['time_bin'] == time_bin].copy()
    
    segments_df = compute_segment_carry(
        futures_at_time, swap_bid, swap_ask, time_bin, 
        min_time_to_expiry=4.0
    )
    if len(segments_df) == 0:  # Skip if all contracts filtered out
        continue
    # ... rest of processing

# Cell 9: Plot with ordinal labels
def ordinal(n):
    return f"{n}{'th' if 11<=n<=13 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

all_segments_df['segment_label'] = all_segments_df['segment_num'].apply(ordinal)

for segment_num in sorted(all_segments_df['segment_num'].unique()):
    segment_data = all_segments_df[all_segments_df['segment_num'] == segment_num]
    segment_label = segment_data['segment_label'].iloc[0]
    
    axes[0].plot(segment_data['time_bin'], segment_data['carry_mid'] * 100, 
                 label=segment_label, linewidth=2)
```

## Backward Compatibility

**Breaking changes**:
1. `compute_segment_carry()` now returns `segment_num` column
2. Removed `carry_lower_pct`, `carry_upper_pct`, `carry_mid_pct` columns (use `* 100` instead)

**Optional parameters** (backward compatible):
- `min_time_to_expiry` defaults to `None` (no filtering)

## Dealing with Negative Carry Rates

You noted seeing negative carry rates due to noise in high/low observations. This happens when:
- One futures contract trades at its daily low
- Another futures contract doesn't trade near that same moment
- Creates artificial spread that looks like backwardation

**Potential solutions** (not implemented yet):
1. Use longer candle periods (e.g., 1H instead of 1min) to smooth noise
2. Apply median filter to carry rates
3. Use volume-weighted bounds instead of simple high/low
4. Detect and clip statistical outliers

For now, the 4-hour expiry filter helps by removing the most volatile segment where this is most problematic.

