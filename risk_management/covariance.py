"""Functions for Functional Tests 1.1 through 2.3.

Test/function map
-----------------
- Tests 1.1-1.4: ``missing_covariance``
- Tests 2.1-2.2: ``ew_covariance`` (with ``ew_weights`` as its helper)
- Test 2.3: ``covariance_with_ew_variance``
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matrix(values: ArrayLike) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=float)
    if out.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if out.shape[0] < 2:
        raise ValueError("at least two observations are required")
    return out


def missing_covariance(
    values: ArrayLike,
    *,
    skip_missing: bool = True,
    correlation: bool = False,
) -> NDArray[np.float64]:
    """Estimate covariance/correlation in the presence of missing values.

    ``skip_missing=True`` performs complete-case deletion.  ``False`` uses the
    observations available for each pair of columns.  Sample covariance
    (``ddof=1``) is used to match the course reference.
    """

    x = _matrix(values)
    estimator = np.corrcoef if correlation else np.cov

    if skip_missing:
        complete = x[~np.isnan(x).any(axis=1)]
        if complete.shape[0] < 2:
            raise ValueError("fewer than two complete observations remain")
        return np.asarray(estimator(complete, rowvar=False), dtype=float)

    n_assets = x.shape[1]
    result = np.empty((n_assets, n_assets), dtype=float)
    for row in range(n_assets):
        for col in range(row + 1):
            available = ~np.isnan(x[:, row]) & ~np.isnan(x[:, col])
            if available.sum() < 2:
                raise ValueError(
                    f"fewer than two paired observations for columns {row} and {col}"
                )
            pair = x[available][:, [row, col]]
            value = estimator(pair, rowvar=False)[0, 1]
            result[row, col] = result[col, row] = value
    return result


def ew_weights(n_observations: int, decay: float) -> NDArray[np.float64]:
    """Return normalized exponential weights, oldest to newest."""

    if n_observations < 1:
        raise ValueError("n_observations must be positive")
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be between 0 and 1")
    powers = np.arange(n_observations - 1, -1, -1, dtype=float)
    weights = (1.0 - decay) * np.power(decay, powers)
    return weights / weights.sum()


def ew_covariance(values: ArrayLike, decay: float) -> NDArray[np.float64]:
    """Exponentially weighted population covariance around the weighted mean."""

    x = _matrix(values)
    if not np.isfinite(x).all():
        raise ValueError("ew_covariance does not accept missing or infinite values")
    weights = ew_weights(x.shape[0], decay)
    demeaned = x - weights @ x
    return (demeaned * np.sqrt(weights)[:, None]).T @ (
        demeaned * np.sqrt(weights)[:, None]
    )


def covariance_with_ew_variance(
    values: ArrayLike,
    *,
    variance_decay: float = 0.97,
    correlation_decay: float = 0.94,
) -> NDArray[np.float64]:
    """Combine EW volatilities and correlations estimated at different decays."""

    variance_cov = ew_covariance(values, variance_decay)
    correlation_cov = ew_covariance(values, correlation_decay)
    vol = np.sqrt(np.diag(variance_cov))
    corr_vol = np.sqrt(np.diag(correlation_cov))
    if np.any(corr_vol <= 0):
        raise ValueError("correlation covariance contains a zero variance")
    corr = correlation_cov / np.outer(corr_vol, corr_vol)
    return corr * np.outer(vol, vol)
