"""
Plotting utilities for forward curve visualization.

Provides clean, reusable plotting functions for forward curves, NS factors,
tracking, and comparison plots.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Tuple, List
from .pchip import PCHIPCurve, reconstruct_forward
from .kalman_ns import NSCarryState, reconstruct_ns_forward


def plot_pchip_snapshot(
    df_pchip: pl.DataFrame,
    snapshot_time: int,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None
) -> plt.Axes:
    """
    Plot a single PCHIP forward curve snapshot with observed pillars.
    
    Args:
        df_pchip: DataFrame with PCHIP curves
        snapshot_time: timeMs to plot
        ax: Optional axes to plot on
        title: Optional custom title
        
    Returns:
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    snapshot = df_pchip.filter(pl.col('timeMs') == snapshot_time)
    swap_points = snapshot.filter(pl.col('source') == 'swap')
    observed = snapshot.filter(pl.col('source') == 'observed')
    
    # Plot bid/ask curves
    ax.plot(snapshot['T'], snapshot['F_bid'], 'b-', label='PCHIP Bid', alpha=0.8, linewidth=2)
    ax.plot(snapshot['T'], snapshot['F_ask'], 'r-', label='PCHIP Ask', alpha=0.8, linewidth=2)
    
    # Mark observed pillars
    ax.scatter(observed['T'], observed['F_bid'], c='blue', marker='o', s=50, 
               zorder=5, label='Observed Bid', edgecolors='darkblue', linewidths=1.5)
    ax.scatter(observed['T'], observed['F_ask'], c='red', marker='o', s=50, 
               zorder=5, label='Observed Ask', edgecolors='darkred', linewidths=1.5)
    
    # Mark swap anchor
    ax.scatter(swap_points['T'], swap_points['F_bid'], c='blue', marker='*', 
               s=250, zorder=5, label='Swap Anchor', edgecolors='darkblue', linewidths=1.5)
    ax.scatter(swap_points['T'], swap_points['F_ask'], c='red', marker='*', 
               s=250, zorder=5, edgecolors='darkred', linewidths=1.5)
    
    ax.set_xlabel('Time to Maturity (years)', fontsize=11)
    ax.set_ylabel('Forward Price (USD)', fontsize=11)
    
    if title is None:
        title = f'PCHIP Forward Curve at {datetime.fromtimestamp(snapshot_time/1000)}'
    ax.set_title(title, fontsize=13)
    
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_kalman_snapshot(
    state: NSCarryState,
    T_grid: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None
) -> plt.Axes:
    """
    Plot a single Kalman NS forward curve snapshot.
    
    Args:
        state: NSCarryState to plot
        T_grid: Optional maturity grid (default: 0 to 2 years)
        ax: Optional axes to plot on
        title: Optional custom title
        
    Returns:
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    if T_grid is None:
        T_grid = np.linspace(0, 2.0, 100)
    
    F_bid_grid = reconstruct_ns_forward(state, T_grid, use_bid=True)
    F_ask_grid = reconstruct_ns_forward(state, T_grid, use_bid=False)
    
    ax.plot(T_grid, F_bid_grid, 'b-', label='Kalman Bid', linewidth=2, alpha=0.8)
    ax.plot(T_grid, F_ask_grid, 'r-', label='Kalman Ask', linewidth=2, alpha=0.8)
    
    # Mark swap anchor
    ax.scatter([0], [state.F_ref_bid], c='blue', marker='*', s=250, 
               zorder=5, label='Swap Anchor', edgecolors='darkblue', linewidths=1.5)
    ax.scatter([0], [state.F_ref_ask], c='red', marker='*', s=250, 
               zorder=5, edgecolors='darkred', linewidths=1.5)
    
    ax.set_xlabel('Time to Maturity (years)', fontsize=11)
    ax.set_ylabel('Forward Price (USD)', fontsize=11)
    
    if title is None:
        title = f'Kalman NS Forward Curve'
    ax.set_title(title, fontsize=13)
    
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_ns_factors(
    df_kalman: pl.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = 'Nelson-Siegel Factors Evolution'
) -> plt.Axes:
    """
    Plot Nelson-Siegel factors over time.
    
    Args:
        df_kalman: DataFrame with Kalman NS states
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    times = df_kalman['timeMs'].to_numpy()
    times_dt = [datetime.fromtimestamp(t/1000) for t in times]
    
    ax.plot(times_dt, df_kalman['beta0'], label='β₀ (Level)', linewidth=2, alpha=0.8)
    ax.plot(times_dt, df_kalman['beta1'], label='β₁ (Slope)', linewidth=2, alpha=0.8)
    ax.plot(times_dt, df_kalman['beta2'], label='β₂ (Curvature)', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Factor Value', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_forward_tracking(
    df_pchip_track: pl.DataFrame,
    df_kalman_track: pl.DataFrame,
    T_track: float,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Axes, dict]:
    """
    Plot forward price tracking comparison between PCHIP and Kalman.
    
    Args:
        df_pchip_track: DataFrame with PCHIP tracking data
        df_kalman_track: DataFrame with Kalman tracking data
        T_track: Maturity being tracked (in years)
        ax: Optional axes to plot on
        
    Returns:
        Tuple of (axes, stats_dict) where stats_dict contains smoothness metrics
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    times_pchip = [datetime.fromtimestamp(t/1000) for t in df_pchip_track['timeMs']]
    times_kalman = [datetime.fromtimestamp(t/1000) for t in df_kalman_track['timeMs']]
    
    ax.plot(times_pchip, df_pchip_track['F_mid'], label='PCHIP', 
            linewidth=2, alpha=0.8, color='steelblue')
    ax.plot(times_kalman, df_kalman_track['F_mid'], label='Kalman NS', 
            linewidth=2, alpha=0.8, color='coral')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Forward Price (USD)', fontsize=12)
    ax.set_title(f'{T_track*12:.1f}-Month Forward Price Evolution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Compute smoothness metrics
    pchip_changes = np.abs(np.diff(np.log(df_pchip_track['F_mid'].to_numpy())))
    kalman_changes = np.abs(np.diff(np.log(df_kalman_track['F_mid'].to_numpy())))
    
    stats = {
        'pchip_mean': float(pchip_changes.mean()),
        'pchip_std': float(pchip_changes.std()),
        'kalman_mean': float(kalman_changes.mean()),
        'kalman_std': float(kalman_changes.std())
    }
    
    return ax, stats


def plot_error_distribution(
    errors: np.ndarray,
    method_name: str,
    ax: Optional[plt.Axes] = None,
    bins: int = 50
) -> plt.Axes:
    """
    Plot error distribution histogram with stats.
    
    Args:
        errors: Array of errors
        method_name: Name of the method (for title)
        ax: Optional axes to plot on
        bins: Number of histogram bins
        
    Returns:
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(errors, bins=bins, alpha=0.7, edgecolor='black', color='steelblue')
    
    # Add vertical lines for mean and median
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    
    ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}')
    ax.axvline(median_err, color='green', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f}')
    
    ax.set_xlabel('Error (USD)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{method_name} - Error Distribution', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_spread_analysis(
    df_curves: pl.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = 'Bid-Ask Spread Analysis'
) -> plt.Axes:
    """
    Plot bid-ask spread over maturity.
    
    Args:
        df_curves: DataFrame with forward curves (must have T, F_bid, F_ask)
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute spread
    df_spread = df_curves.with_columns([
        ((pl.col('F_ask') - pl.col('F_bid')) / pl.col('F_bid') * 10000).alias('spread_bps')
    ])
    
    # Group by T and compute statistics
    spread_stats = df_spread.group_by('T').agg([
        pl.col('spread_bps').mean().alias('mean_spread'),
        pl.col('spread_bps').std().alias('std_spread'),
        pl.col('spread_bps').quantile(0.25).alias('q25_spread'),
        pl.col('spread_bps').quantile(0.75).alias('q75_spread'),
    ]).sort('T')
    
    T_vals = spread_stats['T'].to_numpy()
    mean_vals = spread_stats['mean_spread'].to_numpy()
    std_vals = spread_stats['std_spread'].to_numpy()
    q25_vals = spread_stats['q25_spread'].to_numpy()
    q75_vals = spread_stats['q75_spread'].to_numpy()
    
    ax.plot(T_vals, mean_vals, 'b-', linewidth=2, label='Mean Spread')
    ax.fill_between(T_vals, q25_vals, q75_vals, alpha=0.3, label='25th-75th Percentile')
    
    ax.set_xlabel('Time to Maturity (years)', fontsize=11)
    ax.set_ylabel('Spread (bps)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax


def create_comparison_figure(
    df_pchip: pl.DataFrame,
    df_kalman: pl.DataFrame,
    snapshot_idx: int = 0
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a comprehensive comparison figure with PCHIP, Kalman, and NS factors.
    
    Args:
        df_pchip: DataFrame with PCHIP curves
        df_kalman: DataFrame with Kalman NS states
        snapshot_idx: Index of snapshot to plot
        
    Returns:
        Tuple of (figure, axes_list)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # PCHIP snapshot
    snapshot_time = df_pchip['timeMs'].unique()[snapshot_idx]
    plot_pchip_snapshot(df_pchip, snapshot_time, ax=ax1)
    
    # Kalman snapshot - use convenience constructor
    state = NSCarryState.from_row(df_kalman[snapshot_idx])
    plot_kalman_snapshot(state, ax=ax2)
    
    # NS factors evolution
    plot_ns_factors(df_kalman, ax=ax3)
    
    fig.suptitle(f'Forward Curve Comparison - Snapshot {snapshot_idx}', fontsize=15, y=0.995)
    
    return fig, [ax1, ax2, ax3]

