"""Functions for Functional Tests 1.1 through 2.3.

Test/function map
-----------------
- Tests 1.1-1.4: ``missing_covariance``
- Tests 2.1-2.2: ``ew_covariance`` (with ``ew_weights`` as its helper)
- Test 2.3: ``covariance_with_ew_variance``
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .psd import chol_psd, higham_nearest_psd, near_psd


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


def joint_observation_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count jointly observed rows for every pair of DataFrame columns."""

    observed = frame.notna().astype(int)
    counts = observed.T @ observed
    counts.index.name = "Series"
    return counts


def complete_observation_count(frame: pd.DataFrame) -> int:
    """Count rows on which every DataFrame column is observed."""

    return int(frame.notna().all(axis=1).sum())


def sample_standard_deviations(
    values: ArrayLike, *, skip_missing: bool = False
) -> NDArray[np.float64]:
    """Return column-wise sample standard deviations (``ddof=1``).

    Set ``skip_missing=True`` to estimate every column from all of its finite
    observations.  The default rejects missing and infinite observations so a
    caller cannot silently mix sample sizes.
    """

    x = _matrix(values)
    if skip_missing:
        finite_counts = np.isfinite(x).sum(axis=0)
        if np.any(finite_counts < 2):
            raise ValueError("every column must contain at least two finite observations")
        finite_values = np.where(np.isfinite(x), x, np.nan)
        return np.asarray(np.nanstd(finite_values, axis=0, ddof=1), dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("values must contain only finite observations")
    return np.asarray(np.std(x, axis=0, ddof=1), dtype=float)


def correlation_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    """Estimate complete-case and pairwise correlations and diagnose PSD status."""

    values = frame.to_numpy(dtype=float)
    complete = missing_covariance(values, skip_missing=True, correlation=True)
    pairwise = missing_covariance(values, skip_missing=False, correlation=True)
    output: dict[str, object] = {
        "complete": complete,
        "pairwise": pairwise,
        "complete_eigenvalues": np.linalg.eigvalsh(complete),
        "pairwise_eigenvalues": np.linalg.eigvalsh(pairwise),
    }
    for label, matrix in (("complete", complete), ("pairwise", pairwise)):
        try:
            output[f"{label}_cholesky"] = chol_psd(matrix)
            output[f"{label}_cholesky_status"] = "succeeded"
        except np.linalg.LinAlgError as exc:
            output[f"{label}_cholesky"] = None
            output[f"{label}_cholesky_status"] = f"failed: {exc}"
    return output


def covariance_from_correlation(
    correlation: ArrayLike, standard_deviations: ArrayLike
) -> NDArray[np.float64]:
    """Convert a correlation matrix and marginal standard deviations to covariance."""

    rho = np.asarray(correlation, dtype=float)
    volatility = np.asarray(standard_deviations, dtype=float).reshape(-1)
    if rho.shape != (volatility.size, volatility.size):
        raise ValueError("correlation dimensions must match standard_deviations")
    return rho * np.outer(volatility, volatility)


def portfolio_variance(covariance: ArrayLike, positions: ArrayLike) -> float:
    """Return the variance of a linear portfolio."""

    covariance_matrix = np.asarray(covariance, dtype=float)
    weights = np.asarray(positions, dtype=float).reshape(-1)
    if covariance_matrix.shape != (weights.size, weights.size):
        raise ValueError("covariance dimensions must match positions")
    return float(weights @ covariance_matrix @ weights)


def frobenius_distance(left: ArrayLike, right: ArrayLike) -> float:
    """Return the Frobenius distance between two conformable matrices."""

    left_matrix = np.asarray(left, dtype=float)
    right_matrix = np.asarray(right, dtype=float)
    if left_matrix.shape != right_matrix.shape:
        raise ValueError("matrices must have the same shape")
    return float(np.linalg.norm(left_matrix - right_matrix, ord="fro"))


def repair_correlation_summary(correlation: ArrayLike) -> dict[str, object]:
    """Apply near-PSD and Higham repairs and summarize their movement."""

    original = np.asarray(correlation, dtype=float)
    repaired = {
        "Rebonato-Jackel": near_psd(original),
        "Higham": higham_nearest_psd(original),
    }
    summary: dict[str, object] = {}
    for name, matrix in repaired.items():
        summary[name] = {
            "matrix": matrix,
            "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
            "frobenius_distance": frobenius_distance(matrix, original),
            "movement": matrix - original,
        }
    return summary


def largest_matrix_movements(
    before: ArrayLike,
    after: ArrayLike,
    labels: list[str] | tuple[str, ...],
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Rank unique off-diagonal matrix entries by absolute movement."""

    original = np.asarray(before, dtype=float)
    repaired = np.asarray(after, dtype=float)
    if original.shape != repaired.shape or original.shape[0] != len(labels):
        raise ValueError("matrix shapes and labels must agree")
    rows: list[dict[str, float | str]] = []
    for row, column in combinations(range(len(labels)), 2):
        change = repaired[row, column] - original[row, column]
        rows.append(
            {
                "Pair": f"{labels[row]}-{labels[column]}",
                "Before": original[row, column],
                "After": repaired[row, column],
                "Change": change,
                "Absolute Change": abs(change),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Absolute Change", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def smallest_eigenvector_loadings(
    matrix: ArrayLike, labels: list[str] | tuple[str, ...]
) -> pd.DataFrame:
    """Return loadings for the eigenvector associated with the smallest eigenvalue."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape != (len(labels), len(labels)):
        raise ValueError("matrix must be square with one label per dimension")
    eigenvalues, eigenvectors = np.linalg.eigh((values + values.T) / 2.0)
    loadings = eigenvectors[:, 0]
    return (
        pd.DataFrame(
            {
                "Series": labels,
                "Smallest-Eigenvector Loading": loadings,
                "Absolute Loading": np.abs(loadings),
                "Smallest Eigenvalue": eigenvalues[0],
            }
        )
        .sort_values("Absolute Loading", ascending=False)
        .reset_index(drop=True)
    )
