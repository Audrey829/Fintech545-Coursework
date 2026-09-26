"""Portfolio construction functions for Functional Tests 10.1 through 10.4.

Test/function map
-----------------
- Tests 10.1-10.2: ``risk_parity``
- Tests 10.3-10.4: ``maximize_sharpe_ratio``
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize


def _covariance(covariance: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must be finite")
    return (matrix + matrix.T) / 2.0


def component_standard_deviation(
    weights: ArrayLike, covariance: ArrayLike
) -> NDArray[np.float64]:
    """Euler component contributions to portfolio standard deviation."""

    cov = _covariance(covariance)
    w = np.asarray(weights, dtype=float).reshape(-1)
    variance = float(w @ cov @ w)
    if variance <= 0.0:
        raise ValueError("portfolio variance must be positive")
    return w * (cov @ w) / np.sqrt(variance)


def risk_parity(
    covariance: ArrayLike, risk_budget: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Find long-only weights with component risk proportional to a budget."""

    cov = _covariance(covariance)
    n_assets = cov.shape[0]
    budget = (
        np.ones(n_assets)
        if risk_budget is None
        else np.asarray(risk_budget, dtype=float).reshape(-1)
    )
    if budget.size != n_assets or np.any(budget <= 0.0):
        raise ValueError("risk_budget must contain one positive value per asset")

    inverse_budget = 1.0 / budget

    def objective(weights: NDArray[np.float64]) -> float:
        scaled = inverse_budget * component_standard_deviation(weights, cov)
        return float(1e5 * np.sum((scaled - scaled.mean()) ** 2))

    result = optimize.minimize(
        objective,
        np.full(n_assets, 1.0 / n_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": 1e-14, "maxiter": 10_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"risk-parity optimization failed: {result.message}")
    weights = np.maximum(result.x, 0.0)
    return weights / weights.sum()


def maximize_sharpe_ratio(
    covariance: ArrayLike,
    expected_returns: ArrayLike,
    risk_free_rate: float,
    bounds: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Find fully invested weights that maximize the ex-ante Sharpe ratio."""

    cov = _covariance(covariance)
    means = np.asarray(expected_returns, dtype=float).reshape(-1)
    n_assets = cov.shape[0]
    if means.size != n_assets or not np.isfinite(means).all():
        raise ValueError("expected_returns must contain one finite value per asset")

    if bounds is None:
        limits = [(0.0, 1.0)] * n_assets
    else:
        bound_array = np.asarray(bounds, dtype=float)
        if bound_array.shape != (n_assets, 2):
            raise ValueError("bounds must have shape (n_assets, 2)")
        limits = [tuple(row) for row in bound_array]

    lower = np.array([limit[0] for limit in limits])
    upper = np.array([limit[1] for limit in limits])
    if lower.sum() > 1.0 or upper.sum() < 1.0:
        raise ValueError("bounds do not permit weights summing to one")
    start = lower + (1.0 - lower.sum()) * (upper - lower) / (upper - lower).sum()

    def negative_sharpe(weights: NDArray[np.float64]) -> float:
        variance = float(weights @ cov @ weights)
        if variance <= 0.0:
            return 1e12
        return float(-(weights @ means - risk_free_rate) / np.sqrt(variance))

    result = optimize.minimize(
        negative_sharpe,
        start,
        method="SLSQP",
        bounds=limits,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": 1e-14, "maxiter": 10_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"maximum-Sharpe optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)
