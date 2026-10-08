"""Linear factor-model estimation, diagnostics, and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .simulation import simulate_normal


@dataclass(frozen=True)
class OLSFactorFit:
    """Ordinary-least-squares fit for one asset on one factor."""

    alpha: float
    beta: float
    residuals: NDArray[np.float64]
    residual_standard_deviation: float


def fit_market_model(
    asset_returns: ArrayLike, market_returns: ArrayLike
) -> OLSFactorFit:
    """Fit an intercept-and-beta ordinary least-squares market model."""

    asset = np.asarray(asset_returns, dtype=float).reshape(-1)
    market = np.asarray(market_returns, dtype=float).reshape(-1)
    if asset.size != market.size:
        raise ValueError("asset and market returns must have the same length")
    if asset.size < 3 or not np.isfinite(asset).all() or not np.isfinite(market).all():
        raise ValueError("returns must contain at least three finite observations")
    design = np.column_stack([np.ones(market.size), market])
    coefficients = np.linalg.lstsq(design, asset, rcond=None)[0]
    residuals = asset - design @ coefficients
    return OLSFactorFit(
        alpha=float(coefficients[0]),
        beta=float(coefficients[1]),
        residuals=residuals,
        residual_standard_deviation=float(np.std(residuals, ddof=1)),
    )


def market_model_summary(
    fits: dict[str, OLSFactorFit],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabulate fitted coefficients and the residual correlation matrix."""

    if not fits:
        raise ValueError("fits cannot be empty")
    labels = list(fits)
    table = pd.DataFrame(
        [
            {
                "Asset": label,
                "alpha": fits[label].alpha,
                "beta": fits[label].beta,
                "Residual Standard Deviation": fits[label].residual_standard_deviation,
            }
            for label in labels
        ]
    )
    residuals = np.column_stack([fits[label].residuals for label in labels])
    correlations = np.atleast_2d(np.corrcoef(residuals, rowvar=False))
    return table, pd.DataFrame(correlations, index=labels, columns=labels)


def simulate_market_model(
    market_returns: ArrayLike,
    fits: dict[str, OLSFactorFit],
    *,
    n_samples: int,
    seed: int = 545,
    include_residual_correlation: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate zero-mean market-model returns with configurable residual covariance."""

    market = np.asarray(market_returns, dtype=float).reshape(-1)
    labels = list(fits)
    if not labels:
        raise ValueError("fits cannot be empty")
    residual_matrix = np.column_stack([fits[label].residuals for label in labels])
    residual_covariance = np.atleast_2d(
        np.cov(residual_matrix, rowvar=False, ddof=1)
    )
    if not include_residual_correlation:
        residual_covariance = np.diag(np.diag(residual_covariance))
    full_covariance = np.zeros((len(labels) + 1, len(labels) + 1), dtype=float)
    full_covariance[0, 0] = np.var(market, ddof=1)
    full_covariance[1:, 1:] = residual_covariance
    simulated_components = simulate_normal(
        n_samples,
        full_covariance,
        mean=np.zeros(len(labels) + 1),
        seed=seed,
    )
    simulated_market = simulated_components[:, 0]
    simulated_returns = np.column_stack(
        [
            fits[label].beta * simulated_market
            + simulated_components[:, index + 1]
            for index, label in enumerate(labels)
        ]
    )
    return simulated_returns, residual_covariance
