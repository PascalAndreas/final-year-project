"""Put-call parity helpers shared by recipes and evaluation pipelines."""

from typing import Optional

import numpy as np
import polars as pl


def compute_option_parity_table(
    df_options: pl.DataFrame,
    min_moneyness: float = 0.9,
    max_moneyness: float = 1.1,
) -> pl.DataFrame:
    """Build call/put pairs with fitted/implied forward diagnostics."""
    if df_options.is_empty():
        return pl.DataFrame()

    df_filtered = df_options.filter(
        (pl.col('moneyness') >= min_moneyness) &
        (pl.col('moneyness') <= max_moneyness)
    )
    if df_filtered.is_empty():
        return pl.DataFrame()

    df_calls = df_filtered.filter(pl.col('opt_type') == 'C').select([
        'timeMs', 'expiry', 'strike', 'T', 'moneyness',
        'bid_1_px', 'ask_1_px', 'F_bid', 'F_ask'
    ]).rename({
        'bid_1_px': 'call_bid_1_px',
        'ask_1_px': 'call_ask_1_px',
    })

    df_puts = df_filtered.filter(pl.col('opt_type') == 'P').select([
        'timeMs', 'expiry', 'strike', 'bid_1_px', 'ask_1_px'
    ]).rename({
        'bid_1_px': 'put_bid_1_px',
        'ask_1_px': 'put_ask_1_px',
    })

    df_pairs = df_calls.join(df_puts, on=['timeMs', 'expiry', 'strike'], how='inner')
    if df_pairs.is_empty():
        return pl.DataFrame()

    strikes = df_pairs['strike'].to_numpy()
    call_bids = df_pairs['call_bid_1_px'].to_numpy()
    call_asks = df_pairs['call_ask_1_px'].to_numpy()
    put_bids = df_pairs['put_bid_1_px'].to_numpy()
    put_asks = df_pairs['put_ask_1_px'].to_numpy()
    F_bids = df_pairs['F_bid'].to_numpy()
    F_asks = df_pairs['F_ask'].to_numpy()

    valid_mask = ~(
        np.isnan(call_bids) | np.isnan(call_asks) |
        np.isnan(put_bids) | np.isnan(put_asks) |
        np.isnan(F_bids) | np.isnan(F_asks) |
        (call_bids <= 0) | (call_asks <= 0) |
        (put_bids <= 0) | (put_asks <= 0)
    )
    if not np.any(valid_mask):
        return pl.DataFrame()

    F_implied_bids = strikes + (call_bids - put_asks)
    F_implied_asks = strikes + (call_asks - put_bids)
    F_implied_mids = (F_implied_bids + F_implied_asks) / 2
    F_mids = (F_bids + F_asks) / 2

    error_bids_bps = (np.log(F_implied_bids) - np.log(F_bids)) * 10000
    error_asks_bps = (np.log(F_implied_asks) - np.log(F_asks)) * 10000
    error_mids_bps = (np.log(F_implied_mids) - np.log(F_mids)) * 10000
    call_mid = (call_asks + call_bids) / 2
    put_mid = (put_asks + put_bids) / 2
    call_spread_bps = np.divide(
        (call_asks - call_bids) * 10000,
        call_mid,
        out=np.full_like(call_mid, np.nan, dtype=np.float64),
        where=call_mid > 0,
    )
    put_spread_bps = np.divide(
        (put_asks - put_bids) * 10000,
        put_mid,
        out=np.full_like(put_mid, np.nan, dtype=np.float64),
        where=put_mid > 0,
    )

    return (
        df_pairs.with_columns([
            pl.Series('F_mid', F_mids),
            pl.Series('F_implied_bid', F_implied_bids),
            pl.Series('F_implied_ask', F_implied_asks),
            pl.Series('F_implied_mid', F_implied_mids),
            pl.Series('error_bid_bps', error_bids_bps),
            pl.Series('error_ask_bps', error_asks_bps),
            pl.Series('error_mid_bps', error_mids_bps),
            pl.Series('call_spread_bps', call_spread_bps),
            pl.Series('put_spread_bps', put_spread_bps),
            pl.Series('valid_mask', valid_mask),
        ])
        .rename({'expiry': 'expiry_dt'})
        .filter(pl.col('valid_mask'))
        .drop('valid_mask')
    )


def summarize_option_parity(df_parity: pl.DataFrame) -> pl.DataFrame:
    """Return aggregate stats for a parity diagnostics table."""
    if df_parity.is_empty():
        return pl.DataFrame()

    return df_parity.select([
        pl.count().alias('n_pairs'),
        pl.col('error_mid_bps').mean().alias('error_mid_bps_mean'),
        pl.col('error_mid_bps').std().alias('error_mid_bps_std'),
        pl.col('error_bid_bps').mean().alias('error_bid_bps_mean'),
        pl.col('error_ask_bps').mean().alias('error_ask_bps_mean'),
        pl.col('call_spread_bps').mean().alias('call_spread_bps_mean'),
        pl.col('put_spread_bps').mean().alias('put_spread_bps_mean'),
    ])
