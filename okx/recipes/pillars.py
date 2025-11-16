import polars as pl
from datetime import date, datetime
from typing import Optional

def prepare_pillars(
    store,
    inst_family: str,
    dates: list[date],
    binning: Optional[str] = None,
    min_time_to_expiry_hours: float = 2.0,
    unique_times: Optional[list[int]] = None,
    drop_pillar_idx: Optional[int] = None,
    batch_days: Optional[int] = None,
    anchor: str = 'SWAP',  # Must be 'SWAP' or 'SPOT'
    verbose: bool = False,
) -> pl.LazyFrame:
    """
    Prepare concatenated pillar data from SWAP/SPOT and FUTURES orderbooks.

    Returns a LazyFrame sorted by (timeMs, T) that contains the latest snapshot
    for each instrument at or before every requested timestamp. This keeps the
    function compatible with store.get_derived() caching and avoids Python-side
    looping over symbols/timestamps.

    This function is mature, but there stands two potential improvements:
     - unique_times binning and ff could be pushed upstream to the store.get call, but this would require clever implementation for compatability with batching.
     - previous day's data could be fetched and processed to ensure early bin/timestamps have full coverage.
    """
    if not dates:
        return pl.LazyFrame()
    if binning is not None and unique_times is not None:
        raise ValueError("Provide either 'binning' for time-based binning or 'unique_times' for explicit timestamps, but not both.")
    if verbose:
        start_time = datetime.now()
        strategy = f'{binning} binning' if binning is not None else f'provided timestamps' if unique_times is not None else 'all timestamps'
        print(f"Constructing pillars for {inst_family} with {anchor.lower()} and futures using {strategy}")

    # =============================================================================
    # Step 1: Fetch and preprocess anchor and futures datasets
    # =============================================================================

    # Build feature list (ordered for performance: strip early to reduce columns)
    features = ['trim', 'strip']
    # using forward fill binning to ensure complete grid for small bin sizes
    # Note: bin_ff creates per-symbol grids only within each symbol's trading range,
    # avoiding null rows before symbols started trading. Forward fill still works
    # over gaps within the trading period.
    if binning:
        features.extend(['bin_ff', 'sink_bins'])
    else:
        features.extend(['dedupe'])
    features.extend(['rel_spread', 'tenor', 'log'])

    shared_params = {
        'inst_family': inst_family,
        'dates': dates,
        'depth': 1,
        'features': features,
        'binning': binning,
        'cache_name': f"{'full' if binning is None else binning}_pillars",
        'batch_days': batch_days,
        'verbose': False,
    }

    lf_anchor = store.get(
        inst_type=anchor,
        **shared_params,
    ).sort('timeMs')

    lf_futures = store.get(
        inst_type='FUTURES',
        **shared_params,
    ).sort(['symbol', 'timeMs'])

    if verbose:
        time_1 = datetime.now()
        print(f" - Time taken to fetch {anchor.lower()} and futures: {time_1 - start_time}")

    # =============================================================================
    # Step 2: Align timestamps and apply filters
    # =============================================================================
    
    if binning is None:
        # No binning - need to align snapshots with join_asof
        if unique_times is None:
            # Use union of anchor and futures timeMs
            lf_times = pl.concat([lf_anchor.select('timeMs'), lf_futures.select('timeMs')]).unique().sort('timeMs').lazy()
        else:
            # Use explicitly provided timestamps
            lf_times = pl.DataFrame({'timeMs': sorted(unique_times)}).lazy()
        
        # Align anchor snapshots to target timestamps
        lf_anchor = (
            lf_times.join_asof(
                lf_anchor,
                on='timeMs',
                strategy='backward'
            ).drop('timeMs_right', strict=False)
        )
        
        # Align futures snapshots to target timestamps (per symbol)
        lf_symbol_times = (
            lf_futures.select('symbol').unique()
            .join(lf_times, how='cross')
            .sort(['symbol', 'timeMs'])
        )
        lf_futures = (
            lf_symbol_times.join_asof(
                lf_futures,
                on='timeMs',
                by='symbol',
                strategy='backward'
            ).drop(['symbol_right', 'timeMs_right'], strict=False)
        )
    
    # Apply time-to-expiry filters to futures
    # This prevents: (1) expired contracts, (2) forward-filled stale data in early roll window
    min_ms = min_time_to_expiry_hours * 3600 * 1000
    lf_futures = lf_futures.filter((pl.col('expiry') - pl.col('timeMs')) >= min_ms)
    
    if verbose:
        time_2 = datetime.now()
        print(f" - Time taken to align snapshots: {time_2 - time_1}")

    # =============================================================================
    # Step 3: Concatenate anchor and futures data
    # =============================================================================
    
    # Ensure column ordering matches before concatenation
    pillar_cols = ['symbol', 'timeMs', 'rel_spread', 'expiry', 'T', 'ln_bid_1_px', 'ln_ask_1_px']
    pillars = pl.concat([
        lf_anchor.select(pillar_cols),
        lf_futures.select(pillar_cols)
    ])
    # Filter to timestamps where both anchor and futures data could exist
    # Drop early rows where only one data source exists
    min_anchor_time = lf_anchor.select(pl.col('timeMs').min()).collect().item()
    min_futures_time = lf_futures.select(pl.col('timeMs').min()).collect().item()
    cutoff_time = max(min_anchor_time, min_futures_time)
    pillars = pillars.filter(pl.col('timeMs') >= cutoff_time)

    if verbose:
        time_3 = datetime.now()
        print(f" - Time taken to concatenate and filter: {time_3 - time_2}")
    
    # =============================================================================
    # Step 4: Add pillar indexing and filtering
    # =============================================================================

    pillars = (
        pillars
        .sort(['timeMs', 'T'])
        .with_columns(
            pl.int_range(pl.len()).over('timeMs').alias('pillar_idx')
        )
    )

    if drop_pillar_idx is not None:
        pillars = pillars.filter(pl.col('pillar_idx') != drop_pillar_idx)
    
    if verbose:
        end_time = datetime.now()
        print(f" - Time taken to index{', drop' if drop_pillar_idx is not None else ''} and sort pillars: {end_time - time_3}")
        print(f" - Total time taken to prepare pillars: {end_time - start_time}")
    
    return pillars