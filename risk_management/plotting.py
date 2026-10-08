"""Reusable plots for return, P&L, and dependence diagnostics."""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_return_series(
    returns: pd.Series,
    *,
    regime_start: int | None = None,
):
    """Plot a return series with an optional visible-regime marker."""

    figure, axis = plt.subplots(figsize=(10, 3.6))
    axis.plot(returns.index, returns.to_numpy(), color="#2457A6", linewidth=0.9)
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    if regime_start is not None:
        axis.axvline(regime_start, color="#D97706", linestyle="--", linewidth=1.5)
        axis.text(
            regime_start,
            axis.get_ylim()[1] * 0.88,
            "visible volatility break",
            color="#9A5204",
            ha="right",
            va="top",
        )
    axis.set_title("Demeaned daily arithmetic returns")
    axis.set_xlabel("Trading day")
    axis.set_ylabel("Return")
    axis.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.8)
    figure.tight_layout()
    return figure, axis


def plot_pnl_distributions(
    pnl: pd.DataFrame,
    *,
    columns: tuple[str, str],
):
    """Plot comparable histograms for two P&L series."""

    lower = float(pnl[list(columns)].min().min())
    upper = float(pnl[list(columns)].max().max())
    bins = np.linspace(lower, upper, 65)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
    colors = ("#2457A6", "#D97706")
    for axis, column, color in zip(axes, columns, colors):
        axis.hist(pnl[column], bins=bins, color=color, alpha=0.85, edgecolor="white")
        axis.axvline(0.0, color="#333333", linewidth=0.9)
        axis.set_title(column)
        axis.set_xlabel("One-year P&L ($)")
        axis.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel("Scenario count")
    figure.suptitle("Concentrated versus diversified bond P&L", y=1.02)
    figure.tight_layout()
    return figure, axes


def plot_rank_pairs(uniforms: pd.DataFrame):
    """Plot all pairwise rank-uniform relationships."""

    pairs = list(combinations(uniforms.columns, 2))
    figure, axes = plt.subplots(1, len(pairs), figsize=(12, 3.6), sharex=True, sharey=True)
    for axis, (left, right) in zip(np.atleast_1d(axes), pairs):
        axis.scatter(
            uniforms[left],
            uniforms[right],
            s=11,
            alpha=0.45,
            color="#2457A6",
            edgecolors="none",
        )
        for cutoff in (0.025, 0.975):
            axis.axvline(cutoff, color="#D97706", linestyle="--", linewidth=0.8)
            axis.axhline(cutoff, color="#D97706", linestyle="--", linewidth=0.8)
        axis.set_title(f"{left} vs {right}")
        axis.set_xlabel(f"Rank uniform: {left}")
        axis.grid(color="#D8DEE8", linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel("Rank uniform")
    figure.suptitle("Pairwise ranks show dependence in the corners", y=1.02)
    figure.tight_layout()
    return figure, axes
