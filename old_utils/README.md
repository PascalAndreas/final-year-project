# Utils Module

Utility functions for analyzing Deribit Bitcoin options and futures data.

## Modules

### `api.py` - Data Fetching
Functions to retrieve historical trade data from Deribit's API.

```python
from utils import get_trades, get_trades_by_instrument

# Fetch all BTC options for a time period
options_df = get_trades(
    currency="BTC",
    type="option", 
    start_timestamp=start_ms,
    end_timestamp=end_ms
)

# Fetch trades for a specific instrument
trades_df = get_trades_by_instrument(
    instrument_name="BTC-31OCT25-120000-C",
    start_timestamp=start_ms,
    end_timestamp=end_ms
)
```

### `black.py` - Black-76 Model
Black-76 futures option pricing model implementation.

```python
from utils import black76_implied_volatility

# Calculate implied volatility from market price
iv = black76_implied_volatility(
    market_price=0.05,      # Observed option price
    F=110000,               # Forward/futures price  
    K=120000,               # Strike price
    T=0.0192,               # Time to expiry (years)
    r=0.0,                  # Risk-free rate
    option_type='C'         # 'C' for call, 'P' for put
)
```

### `helpers.py` - Utility Functions
Helper functions for options calculations.

```python
from utils import moneyness, parse_option_instrument

# Calculate forward moneyness (K/F)
m = moneyness(strike=120000, forward=110000)  # Returns 1.0909

# Parse instrument name
expiry, strike, opt_type = parse_option_instrument("BTC-31OCT25-120000-C")
# Returns: ('31OCT25', 120000, 'C')
```

### `iv.py` - IV Surface Visualization
Interactive 3D implied volatility surface viewer with time slider.

```python
from utils import prepare_iv_timeseries, create_iv_surface_viewer

# Step 1: Prepare IV timeseries
iv_timeseries = prepare_iv_timeseries(
    options_df=options_df,
    futures_df=futures_df,
    time_step_minutes=5,      # Snapshot every 5 minutes
    lookback_seconds=21600,   # 6 hour cutoff for stale data
    risk_free_rate=0.0
)

# Step 2: Create interactive viewer
fig = create_iv_surface_viewer(
    iv_timeseries=iv_timeseries,
    options_df=options_df,
    futures_df=futures_df,
    trade_window_minutes=10,
    show_trades=True
)

# Step 3: Display
fig.show()
```

#### IV Surface Features

- **Time Slider**: Navigate through time to see IV surface evolution
- **Separate Subplots**: Calls (left) and Puts (right)
- **Surface Interpolation**: Smooth cubic interpolation of sparse data
- **Staleness Indicators**: Data points colored by age (green=fresh, red=stale)
- **Trade Overlay**: Recent trades shown as scatter points
- **Log-Moneyness**: X-axis uses ln(K/F) for better symmetry
- **6-Hour Cutoff**: Hard cutoff excludes marks older than 6 hours

#### Additional IV Functions

```python
from utils import compute_iv_at_timestamp, log_moneyness

# Get IV surface at a specific timestamp
iv_df = compute_iv_at_timestamp(
    options_df=options_df,
    futures_df=futures_df,
    timestamp=timestamp_ms,
    lookback_seconds=21600
)

# Calculate log-moneyness
log_m = log_moneyness(strike=120000, forward=110000)  # Returns ln(K/F)
```

## Data Requirements

All functions expect DataFrames with the following structure:

### Options DataFrame
Required columns from Deribit API:
- `timestamp`: Trade timestamp (milliseconds)
- `instrument_name`: Full instrument name (e.g., "BTC-31OCT25-120000-C")
- `price`: Trade price
- `mark_price`: Mark price at trade time
- `direction`: 'buy' or 'sell'
- `amount`: Trade size (contracts)
- Additional columns preserved as-is

### Futures DataFrame
Required columns from Deribit API:
- `timestamp`: Trade timestamp (milliseconds)
- `instrument_name`: Full instrument name (e.g., "BTC-31OCT25")
- `mark_price`: Mark price at trade time
- Additional columns preserved as-is

## Installation Requirements

```bash
pip install numpy scipy pandas plotly tqdm requests
```

## Example Workflow

See `iv_surface_example.py` in the project root for a complete example.

```python
from datetime import datetime
from utils import (
    get_trades, 
    prepare_iv_timeseries, 
    create_iv_surface_viewer
)

# 1. Fetch data
start = int(datetime(2025, 10, 13).timestamp() * 1000)
end = int(datetime(2025, 10, 18).timestamp() * 1000)

options_df = get_trades("BTC", "option", start, end)
futures_df = get_trades("BTC", "future", start, end)

# 2. Compute IV surfaces
iv_timeseries = prepare_iv_timeseries(
    options_df, futures_df, 
    time_step_minutes=5,
    lookback_seconds=21600
)

# 3. Visualize
fig = create_iv_surface_viewer(
    iv_timeseries, options_df, futures_df,
    trade_window_minutes=10,
    show_trades=True
)

fig.show()
```

## Notes

- **Performance**: Pre-computing IV timeseries can take several minutes for large datasets. Progress bars show status.
- **Staleness**: The 6-hour lookback window balances data freshness vs availability. Adjust `lookback_seconds` if needed.
- **Interpolation**: Cubic interpolation is used for smooth surfaces but may extrapolate unrealistically in sparse regions.
- **Memory**: Large datasets with fine time steps may require significant memory. Consider sampling if needed.

