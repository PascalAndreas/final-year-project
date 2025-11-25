"""
Plotting functions for polars LazyFrames.
"""
from typing import Optional, Union
from datetime import datetime

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_histogram(
    lf: pl.LazyFrame,
    col: Union[str, list[str]],
    ax: Optional[plt.Axes] = None,
    bins: int = 50,
    **kwargs
) -> plt.Axes:
    """
    Plot histogram of a column with statistics.
    
    Args:
        lf: LazyFrame with column(s) to plot
        col: Column name(s) to plot. If list, plots multiple histograms
        ax: Optional axes to plot on
        bins: Number of histogram bins
        **kwargs: Additional kwargs passed to ax.hist
        
    Returns:
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect data
    cols = [col] if isinstance(col, str) else col
    df = lf.select(['timeMs'] + cols).collect()
    
    # Plot histogram(s)
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
    for i, col in enumerate(cols):
        values = df[col].to_numpy()
        values = values[~np.isnan(values)]  # Remove NaN values
        
        color = kwargs.pop('color', colors[i % len(colors)])
        alpha = kwargs.pop('alpha', 0.6 if len(cols) > 1 else 0.7)
        
        ax.hist(values, bins=bins, alpha=alpha, edgecolor='black', 
                color=color, label=col, **kwargs)
        
        # Add statistics for single column
        if len(cols) == 1:
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                      label=f'Median: {median_val:.2f}')
            
            # Add text box with stats
            stats_text = f'Std: {std_val:.2f}\nN: {len(values):,}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(cols[0] if len(cols) == 1 else 'Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{cols[0]} Distribution' if len(cols) == 1 else 'Distribution')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_mean_spread_over_time(
    lf: pl.LazyFrame,
    col: Union[str, list[str]],
    ax: Optional[plt.Axes] = None,
    resample_freq: str = '1h',
    show_spread: bool = True,
    spread_quantiles: tuple[float, float] = (0.25, 0.75),
    include_zero: bool = True,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot mean and spread over time.
    
    Args:
        lf: LazyFrame with column(s) and timeMs
        col: Column name(s) to plot. If list, plots multiple lines
        ax: Optional axes to plot on
        resample_freq: Frequency to resample data (e.g., '1h', '30m', '1d')
        show_spread: Whether to show spread as shaded region
        spread_quantiles: Quantiles to use for spread (default: 25th-75th percentile)
        include_zero: Whether to include zero in y-axis (default: True)
        title: Optional title for the plot
        ylabel: Optional y-axis label
        **kwargs: Additional kwargs passed to ax.plot
        
    Returns:
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    # Collect data
    cols = [col] if isinstance(col, str) else col
    df = lf.select(['timeMs'] + cols).collect()
    
    # Convert timeMs to datetime
    df = df.with_columns([
        pl.col('timeMs').cast(pl.Datetime('ms')).alias('time')
    ])
    
    # Resample and compute statistics
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
    
    for i, col in enumerate(cols):
        # Group by time bins and compute stats
        df_resampled = (
            df.with_columns([
                pl.col('time').dt.truncate(resample_freq).alias('time_bin')
            ])
            .group_by('time_bin')
            .agg([
                pl.col(col).mean().alias('mean'),
                pl.col(col).std().alias('std'),
                pl.col(col).quantile(spread_quantiles[0]).alias('lower'),
                pl.col(col).quantile(spread_quantiles[1]).alias('upper'),
                pl.col(col).count().alias('count')
            ])
            .sort('time_bin')
        )
        
        time_vals = df_resampled['time_bin'].to_numpy()
        mean_vals = df_resampled['mean'].to_numpy()
        lower_vals = df_resampled['lower'].to_numpy()
        upper_vals = df_resampled['upper'].to_numpy()
        
        color = kwargs.pop('color', colors[i % len(colors)])
        label = kwargs.pop('label', col)
        
        # Plot mean
        ax.plot(time_vals, mean_vals, color=color, linewidth=2, 
               label=label, **kwargs)
        
        # Plot spread
        if show_spread:
            ax.fill_between(time_vals, lower_vals, upper_vals, 
                          alpha=0.25, color=color,
                          label=f'{label} ({int(spread_quantiles[0]*100)}-{int(spread_quantiles[1]*100)}%ile)')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(ylabel or (cols[0] if len(cols) == 1 else 'Value'), fontsize=11)
    ax.set_title(title or (f'{cols[0]} Over Time' if len(cols) == 1 else 'Values Over Time'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at zero if requested
    if include_zero:
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    return ax


def plot_dual_axis_over_time(
    lf: pl.LazyFrame,
    expr1: pl.Expr,
    expr2: pl.Expr,
    label1: str,
    label2: str,
    ax: Optional[plt.Axes] = None,
    resample_freq: str = '1h',
    show_spread: bool = True,
    spread_quantiles: tuple[float, float] = (0.25, 0.75),
    color1: str = 'steelblue',
    color2: str = 'coral',
) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot two expressions over time with dual y-axes.
    
    Useful for plotting variables with different orders of magnitude.
    
    Args:
        lf: LazyFrame with timeMs column
        expr1: Polars expression for left y-axis
        expr2: Polars expression for right y-axis
        label1: Label for first expression
        label2: Label for second expression
        ax: Optional axes to plot on (will be left axis)
        resample_freq: Frequency to resample data (e.g., '1h', '30m', '1d')
        show_spread: Whether to show spread as shaded region
        spread_quantiles: Quantiles to use for spread
        color1: Color for first expression
        color2: Color for second expression
        
    Returns:
        Tuple of (left_axis, right_axis)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    # Collect data with both expressions
    df = lf.select([
        'timeMs',
        expr1.alias('val1'),
        expr2.alias('val2'),
    ]).collect()
    
    # Convert timeMs to datetime
    df = df.with_columns([
        pl.col('timeMs').cast(pl.Datetime('ms')).alias('time')
    ])
    
    # Resample and compute statistics for both expressions
    df_resampled = (
        df.with_columns([
            pl.col('time').dt.truncate(resample_freq).alias('time_bin')
        ])
        .group_by('time_bin')
        .agg([
            pl.col('val1').mean().alias('mean1'),
            pl.col('val1').quantile(spread_quantiles[0]).alias('lower1'),
            pl.col('val1').quantile(spread_quantiles[1]).alias('upper1'),
            pl.col('val2').mean().alias('mean2'),
            pl.col('val2').quantile(spread_quantiles[0]).alias('lower2'),
            pl.col('val2').quantile(spread_quantiles[1]).alias('upper2'),
        ])
        .sort('time_bin')
    )
    
    time_vals = df_resampled['time_bin'].to_numpy()
    
    # Plot first expression on left axis
    ax.plot(time_vals, df_resampled['mean1'].to_numpy(), 
           color=color1, linewidth=2, label=label1)
    if show_spread:
        ax.fill_between(time_vals, 
                       df_resampled['lower1'].to_numpy(),
                       df_resampled['upper1'].to_numpy(),
                       alpha=0.25, color=color1)
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(label1, fontsize=11, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.grid(True, alpha=0.3)
    
    # Create second y-axis and plot second expression
    ax2 = ax.twinx()
    ax2.plot(time_vals, df_resampled['mean2'].to_numpy(),
            color=color2, linewidth=2, label=label2)
    if show_spread:
        ax2.fill_between(time_vals,
                        df_resampled['lower2'].to_numpy(),
                        df_resampled['upper2'].to_numpy(),
                        alpha=0.25, color=color2)
    
    ax2.set_ylabel(label2, fontsize=11, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    return ax, ax2


def plot_over_time_by_group(
    lf: pl.LazyFrame,
    col: str,
    group_col: str,
    ax: Optional[plt.Axes] = None,
    resample_freq: Optional[str] = None,
    n_bins: Optional[int] = None,
    show_spread: bool = False,
    spread_quantiles: tuple[float, float] = (0.25, 0.75),
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_legend: bool = True,
    color_by: Optional[str] = None,
    cmap: str = 'viridis',
    **kwargs
) -> plt.Axes:
    """
    Plot column values over time grouped by a categorical or continuous column.
    
    Handles two use cases:
    - Discrete groups (e.g., pillar_idx): plots each group as separate line
    - Continuous values (e.g., moneyness): bins into n_bins groups first
    
    Args:
        lf: LazyFrame with column, group column, and timeMs
        col: Column to plot
        group_col: Column to group by
        ax: Optional axes to plot on
        resample_freq: Frequency to resample data (e.g., '1h', '30m', '1d'). If None, no resampling
        n_bins: If provided, bins continuous values into n_bins groups (e.g., 5 for quintiles)
        show_spread: Whether to show spread as shaded region for each group
        spread_quantiles: Quantiles to use for spread (only used if resampling)
        title: Optional title for the plot
        ylabel: Optional y-axis label
        show_legend: Whether to show legend (default: True)
        color_by: Optional column name to use for continuous color mapping
        cmap: Colormap to use for color_by (default: 'viridis')
        **kwargs: Additional kwargs passed to ax.plot
        
    Returns:
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    # Collect data
    cols_to_select = ['timeMs', col, group_col]
    if color_by is not None and color_by not in cols_to_select:
        cols_to_select.append(color_by)
    df = lf.select(cols_to_select).collect()
    
    # Convert timeMs to datetime
    df = df.with_columns([
        pl.col('timeMs').cast(pl.Datetime('ms')).alias('time')
    ])
    
    # Bin continuous values if requested
    if n_bins is not None:
        df = df.with_columns([
            pl.col(group_col).qcut(n_bins, labels=[f'Q{i+1}' for i in range(n_bins)]).alias('group')
        ])
        group_col_to_use = 'group'
    else:
        group_col_to_use = group_col
    
    # Get unique groups and sort them
    groups = sorted(df[group_col_to_use].unique().to_list())
    
    # Color palette
    if color_by is not None:
        # Compute color value for each group
        color_values = {}
        for group in groups:
            df_group = df.filter(pl.col(group_col_to_use) == group)
            # Use mean value of color_by column for this group
            color_values[group] = df_group[color_by].mean()
        
        # Normalize color values to [0, 1] for colormap
        min_val = min(color_values.values())
        max_val = max(color_values.values())
        val_range = max_val - min_val if max_val != min_val else 1
        
        colormap = plt.cm.get_cmap(cmap)
        colors = {group: colormap((color_values[group] - min_val) / val_range) 
                  for group in groups}
    else:
        # Use discrete color palette
        if len(groups) <= 10:
            color_list = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        else:
            color_list = plt.cm.viridis(np.linspace(0, 1, len(groups)))
        colors = {group: color_list[i] for i, group in enumerate(groups)}
    
    # Plot each group
    for group in groups:
        df_group = df.filter(pl.col(group_col_to_use) == group)
        
        color = kwargs.pop('color', colors[group])
        label = kwargs.pop('label', f'{group_col_to_use}={group}') if show_legend else None
        
        if resample_freq is not None:
            # Resample and compute statistics
            df_resampled = (
                df_group.with_columns([
                    pl.col('time').dt.truncate(resample_freq).alias('time_bin')
                ])
                .group_by('time_bin')
                .agg([
                    pl.col(col).mean().alias('mean'),
                    pl.col(col).quantile(spread_quantiles[0]).alias('lower'),
                    pl.col(col).quantile(spread_quantiles[1]).alias('upper'),
                    pl.col(col).count().alias('count')
                ])
                .sort('time_bin')
            )
            
            time_vals = df_resampled['time_bin'].to_numpy()
            mean_vals = df_resampled['mean'].to_numpy()
            
            # Plot mean
            ax.plot(time_vals, mean_vals, color=color, linewidth=2, 
                   label=label, alpha=0.8, **kwargs)
            
            # Plot spread if requested
            if show_spread:
                lower_vals = df_resampled['lower'].to_numpy()
                upper_vals = df_resampled['upper'].to_numpy()
                ax.fill_between(time_vals, lower_vals, upper_vals, 
                              alpha=0.15, color=color)
        else:
            # No resampling - plot raw data
            df_sorted = df_group.sort('time')
            time_vals = df_sorted['time'].to_numpy()
            col_vals = df_sorted[col].to_numpy()
            
            # Plot raw values
            ax.plot(time_vals, col_vals, color=color, linewidth=1.5, 
                   label=label, alpha=0.8, **kwargs)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(ylabel or col, fontsize=11)
    ax.set_title(title or f'{col} by {group_col_to_use}')
    
    # Add legend if requested
    if show_legend:
        ax.legend(fontsize=9, ncol=min(3, len(groups)))
    
    # Add colorbar if coloring by continuous variable
    if color_by is not None and not show_legend:
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                   norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(color_by, fontsize=11)
    
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at zero
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    return ax