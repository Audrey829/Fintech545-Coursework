"""Reusable descriptive-statistics and regime-diagnostic utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats


def sample_moments(values: ArrayLike) -> dict[str, float]:
    """Return mean, sample variance, bias-corrected skewness, and excess kurtosis."""

    sample = np.asarray(values, dtype=float).reshape(-1)
    sample = sample[np.isfinite(sample)]
    if sample.size < 4:
        raise ValueError("at least four finite observations are required")
    return {
        "Mean": float(np.mean(sample)),
        "Variance": float(np.var(sample, ddof=1)),
        "Skewness": float(stats.skew(sample, bias=False)),
        "Excess Kurtosis": float(stats.kurtosis(sample, fisher=True, bias=False)),
    }


def moments_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute common sample moments for every numeric DataFrame column."""

    return pd.DataFrame({column: sample_moments(frame[column]) for column in frame})


def regime_volatility(values: ArrayLike, *, recent_days: int) -> pd.DataFrame:
    """Compare sample volatility before and during a recent regime."""

    sample = np.asarray(values, dtype=float).reshape(-1)
    if not 1 <= recent_days < sample.size:
        raise ValueError("recent_days must leave observations in both regimes")
    return pd.DataFrame(
        [
            {
                "Regime": "Earlier",
                "Observations": sample.size - recent_days,
                "Standard Deviation": np.std(sample[:-recent_days], ddof=1),
            },
            {
                "Regime": "Recent",
                "Observations": recent_days,
                "Standard Deviation": np.std(sample[-recent_days:], ddof=1),
            },
        ]
    )


def normal_scale_mixture_excess_kurtosis(regimes: pd.DataFrame) -> float:
    """Return theoretical excess kurtosis for a zero-mean Normal scale mixture."""

    required = {"Observations", "Standard Deviation"}
    if not required.issubset(regimes.columns):
        raise ValueError(f"regimes must contain columns {sorted(required)}")
    counts = regimes["Observations"].to_numpy(dtype=float)
    standard_deviations = regimes["Standard Deviation"].to_numpy(dtype=float)
    weights = counts / counts.sum()
    mixture_variance = np.sum(weights * standard_deviations**2)
    fourth_moment = 3.0 * np.sum(weights * standard_deviations**4)
    return float(fourth_moment / mixture_variance**2 - 3.0)


def kurtosis_outlier_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure skewness and kurtosis sensitivity to the most influential deletion."""

    rows = []
    for column in frame:
        sample = frame[column].to_numpy(dtype=float)
        full = sample_moments(sample)
        leave_one_out = [
            sample_moments(np.delete(sample, index)) for index in range(sample.size)
        ]
        kurtoses = np.array([item["Excess Kurtosis"] for item in leave_one_out])
        most_influential = int(
            np.argmax(np.abs(kurtoses - full["Excess Kurtosis"]))
        )
        without = leave_one_out[most_influential]
        rows.append(
            {
                "Series": column,
                "Full Skewness": full["Skewness"],
                "Full Excess Kurtosis": full["Excess Kurtosis"],
                "Most Influential Row Index": most_influential,
                "Observation": sample[most_influential],
                "Skewness Without It": without["Skewness"],
                "Excess Kurtosis Without It": without["Excess Kurtosis"],
                "Kurtosis Absolute Change": abs(
                    without["Excess Kurtosis"] - full["Excess Kurtosis"]
                ),
            }
        )
    return pd.DataFrame(rows)
