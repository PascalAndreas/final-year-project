import polars as pl
from datetime import date, datetime
from typing import Optional

from okx.recipes.helpers import early_roll, finalize_binning

# =============================================================================
# Pillar preparation
# =============================================================================

def prepare_pillars(
    store,
    inst_family: str,
    dates: list[date],
    binning: Optional[str] = None,
    min_time_to_expiry_hours: float = 2.0,
    unique_times: Optional[list[int]] = None,
    drop_pillar_idx: Optional[int] = None,
    batch_days: Optional[int] = None,
    verbose: bool = False,
) -> pl.LazyFrame:
    """
    Prepare concatenated pillar data from SWAP and FUTURES orderbooks.

    Returns a LazyFrame sorted by (timeMs, T) that contains the latest snapshot
    for each instrument at or before every requested timestamp. This keeps the
    function compatible with store.get_derived() caching and avoids Python-side
    looping over symbols/timestamps.
    """
    if not dates:
        return pl.DataFrame().lazy()

    # =============================================================================
    # Step 1: Fetch and preprocess swap and futures datasets
    # =============================================================================
    start_time = datetime.now()
    # Build feature list (ordered for performance: strip early to reduce columns)
    features_base = ['trim', 'strip']
    # using forward fill binning to ensure complete grid for small bin sizes
    # Note: bin_ff creates per-symbol grids only within each symbol's trading range,
    # avoiding null rows before symbols started trading. Forward fill still works
    # over gaps within the trading period.
    if binning:
        features_base.extend(['bin_ff', finalize_binning])
    features_base.extend(['rel_spread', 'tenor', 'log'])

    features_swap = [*features_base]
    features_futures = [*features_base, early_roll(min_time_to_expiry_hours)]

    shared_params = {
        'inst_family': inst_family,
        'dates': dates,
        'depth': 1,
        'binning': binning,
        'cache_name': f"{'full' if binning is None else binning}_pillars",
        'batch_days': batch_days,
        'verbose': False,
    }

    lf_swap = store.get(
        inst_type='SWAP',
        features=features_swap,
        **shared_params,
    ).sort('timeMs')

    lf_futures = store.get(
        inst_type='FUTURES',
        features=features_futures,
        **shared_params,
    ).sort(['symbol', 'timeMs'])

    time_1 = datetime.now()
    if verbose:
        print(f"Time taken to fetch swap and futures: {time_1 - start_time}")

    if unique_times is not None:
        lf_times = pl.DataFrame({'timeMs': sorted(unique_times)}).lazy()
    else:
        lf_times = (
            pl.concat([lf_swap.select('timeMs'), lf_futures.select('timeMs')])
            .unique().sort('timeMs').lazy()
        )

    # Align swap snapshots to requested timestamps (latest observation <= timeMs)
    lf_swap_snapshots = (
        lf_times.join_asof(
            lf_swap,
            on='timeMs',
            strategy='backward',
            suffix='_hist',
        )
        .drop('timeMs_hist', strict=False)
    )

    lf_valid_times = (
        lf_swap_snapshots.select('timeMs')
        .unique()
        .sort('timeMs')
    )

    # Align each futures symbol independently using cross join + asof
    lf_symbol_times = (
        lf_futures.select('symbol')
        .unique()
        .join(lf_valid_times, how='cross')
        .sort(['symbol', 'timeMs'])
    )
    lf_futures_snapshots = (
        lf_symbol_times.join_asof(
            lf_futures,
            on='timeMs',
            by='symbol',
            strategy='backward',
            suffix='_hist',
        )
        .drop(['symbol_hist', 'timeMs_hist'], strict=False)
        .filter(pl.col('expiry') > pl.col('timeMs'))
    )
    
    if verbose:
        # Show data quality AFTER filtering nulls
        time_2 = datetime.now()
        print(f"Time taken to align snapshots: {time_2 - time_1}")

    # Ensure consistent column order before concat (LazyFrames don't have .columns)
    common_cols = ['timeMs', 'symbol', 'rel_spread', 'expiry', 'T', 
                   'ln_bid_1_px', 'ln_ask_1_px']

    pillars = pl.concat([
        lf_swap_snapshots.select(common_cols),
        lf_futures_snapshots.select(common_cols),
    ])
    # =============================================================================
    # Step 2: 
    # =============================================================================

    pillars = (
        pillars
        .sort(['timeMs', 'T'])
        .with_columns([
            pl.int_range(pl.len()).over('timeMs').alias('pillar_idx'),
            pl.len().over('timeMs').alias('_pillars_per_time'),
        ])
    )

    if drop_pillar_idx is not None:
        pillars = pillars.filter(
            pl.col('_pillars_per_time') > drop_pillar_idx
        ).filter(pl.col('pillar_idx') != drop_pillar_idx)

    return (
        pillars
        .drop('_pillars_per_time')
        .sort(['timeMs', 'T'])
    )