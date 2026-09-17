"""Functions for Functional Tests 3.1 through 4.1.

Test/function map
-----------------
- Tests 3.1-3.2: ``near_psd``
- Tests 3.3-3.4: ``higham_nearest_psd``
- Test 4.1: ``chol_psd``
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import sqrtm


def _square_symmetric(matrix: ArrayLike) -> NDArray[np.float64]:
    out = np.asarray(matrix, dtype=float)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError("matrix must be square")
    if not np.isfinite(out).all():
        raise ValueError("matrix must contain only finite values")
    if not np.allclose(out, out.T, rtol=1e-10, atol=1e-12):
        raise ValueError("matrix must be symmetric")
    return (out + out.T) / 2.0


def _as_correlation(matrix: NDArray[np.float64]):
    variances = np.diag(matrix)
    if np.any(variances <= 0):
        raise ValueError("matrix diagonal must be positive")
    is_correlation = np.allclose(variances, 1.0)
    if is_correlation:
        return matrix.copy(), None
    std = np.sqrt(variances)
    return matrix / np.outer(std, std), std


def near_psd(matrix: ArrayLike, epsilon: float = 0.0) -> NDArray[np.float64]:
    """Repair a covariance/correlation matrix by clipping its eigenvalues."""

    original = _square_symmetric(matrix)
    corr, std = _as_correlation(original)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    scale = 1.0 / np.sqrt((eigenvectors * eigenvectors) @ eigenvalues)
    root = np.diag(scale) @ eigenvectors @ np.diag(np.sqrt(eigenvalues))
    repaired = root @ root.T
    if std is not None:
        repaired = repaired * np.outer(std, std)
    return (repaired + repaired.T) / 2.0


def _psd_projection(
    matrix: NDArray[np.float64], weight: NDArray[np.float64]
) -> NDArray[np.float64]:
    weight_root = np.real_if_close(sqrtm(weight)).astype(float)
    inverse_root = np.linalg.inv(weight_root)
    weighted = weight_root @ matrix @ weight_root
    eigenvalues, eigenvectors = np.linalg.eigh((weighted + weighted.T) / 2.0)
    positive = eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return inverse_root @ positive @ inverse_root


def _weighted_norm(
    matrix: NDArray[np.float64], weight: NDArray[np.float64]
) -> float:
    weight_root = np.real_if_close(sqrtm(weight)).astype(float)
    weighted = weight_root @ matrix @ weight_root
    return float(np.sum(weighted * weighted))


def higham_nearest_psd(
    matrix: ArrayLike,
    weight: ArrayLike | None = None,
    *,
    epsilon: float = 1e-9,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> NDArray[np.float64]:
    """Find a nearest PSD correlation/covariance matrix using Higham's method."""

    original = _square_symmetric(matrix)
    corr, std = _as_correlation(original)
    n_assets = corr.shape[0]
    w = np.eye(n_assets) if weight is None else _square_symmetric(weight)

    y = corr.copy()
    reference = corr.copy()
    delta_s = np.zeros_like(corr)
    prior_norm = np.inf

    for _ in range(max_iterations):
        r = y - delta_s
        x = _psd_projection(r, w)
        delta_s = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        current_norm = _weighted_norm(y - reference, w)
        smallest_eigenvalue = np.linalg.eigvalsh(y).min()
        if (
            abs(current_norm - prior_norm) < tolerance
            and smallest_eigenvalue > -epsilon
        ):
            break
        prior_norm = current_norm

    if std is not None:
        y = y * np.outer(std, std)
    return (y + y.T) / 2.0


def chol_psd(matrix: ArrayLike, epsilon: float = -1e-8) -> NDArray[np.float64]:
    """Lower-triangular Cholesky-like root for a PSD matrix."""

    a = _square_symmetric(matrix)
    n_assets = a.shape[0]
    root = np.zeros_like(a)

    for column in range(n_assets):
        diagonal = a[column, column] - root[column, :column] @ root[column, :column]
        if epsilon <= diagonal <= 0.0:
            diagonal = 0.0
        if diagonal < 0.0:
            raise np.linalg.LinAlgError("matrix is not positive semidefinite")
        root[column, column] = np.sqrt(diagonal)
        if root[column, column] == 0.0:
            continue
        for row in range(column + 1, n_assets):
            prior = root[row, :column] @ root[column, :column]
            root[row, column] = (a[row, column] - prior) / root[column, column]
    return root
