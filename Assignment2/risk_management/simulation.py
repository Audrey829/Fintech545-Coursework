"""Functions for Functional Tests 5.1 through 5.5.

Test/function map
-----------------
- Tests 5.1-5.4: ``simulate_normal``
- Test 5.5: ``simulate_pca``
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .psd import chol_psd, near_psd


def _mean_vector(mean: ArrayLike | None, n_assets: int) -> NDArray[np.float64]:
    if mean is None:
        return np.zeros(n_assets)
    result = np.asarray(mean, dtype=float).reshape(-1)
    if result.size != n_assets:
        raise ValueError("mean length must match the covariance dimension")
    return result


def simulate_normal(
    n_samples: int,
    covariance: ArrayLike,
    *,
    mean: ArrayLike | None = None,
    seed: int = 1234,
    fix_method: Callable[[ArrayLike], NDArray[np.float64]] = near_psd,
) -> NDArray[np.float64]:
    """Simulate multivariate Normal observations from a covariance matrix."""

    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    try:
        root = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        try:
            root = chol_psd(cov)
        except np.linalg.LinAlgError:
            root = chol_psd(fix_method(cov))

    rng = np.random.default_rng(seed)
    standard_normal = rng.standard_normal((n_samples, cov.shape[0]))
    return standard_normal @ root.T + _mean_vector(mean, cov.shape[0])


def simulate_pca(
    covariance: ArrayLike,
    n_samples: int,
    *,
    explained_variance: float = 1.0,
    mean: ArrayLike | None = None,
    seed: int = 1234,
) -> NDArray[np.float64]:
    """Simulate using the leading covariance eigenvectors.

    ``explained_variance`` is the minimum cumulative fraction to retain.
    """

    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1]")

    eigenvalues, eigenvectors = np.linalg.eigh((cov + cov.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues >= 1e-8
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    if eigenvalues.size == 0:
        raise ValueError("covariance has no positive eigenvalues")

    if explained_variance < 1.0:
        cumulative = np.cumsum(eigenvalues) / np.trace(cov)
        retained = int(np.searchsorted(cumulative, explained_variance) + 1)
        eigenvalues = eigenvalues[:retained]
        eigenvectors = eigenvectors[:, :retained]

    root = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n_samples, eigenvalues.size))
    return factors @ root.T + _mean_vector(mean, cov.shape[0])
