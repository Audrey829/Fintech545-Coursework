"""Multivariate t and copula functions for Functional Tests 13.1-13.9.

Test/function map
-----------------
- Test 13.1: ``kendall_correlation``
- Tests 13.2-13.4: ``fit_multivariate_t``
- Tests 13.5-13.7: Gaussian/t copula likelihood and information criteria
- Test 13.8: ``tail_dependence_t``
- Test 13.9: ``simulate_t_copula`` (then portfolio risk in risk_metrics.py)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .psd import higham_nearest_psd
from .simulation import simulate_pca


def fix_correlation(
    correlation: ArrayLike, *, tolerance: float = 1e-8, ridge: float = 1e-8
) -> NDArray[np.float64]:
    """Repair a correlation matrix when needed and ensure it is positive definite."""

    result = np.asarray(correlation, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError("correlation must be square")
    if np.linalg.eigvalsh(result).min() < -tolerance:
        result = higham_nearest_psd(result)
    result = (result + result.T) / 2.0
    if np.linalg.eigvalsh(result).min() < ridge:
        result = (result + ridge * np.eye(result.shape[0])) / (1.0 + ridge)
    np.fill_diagonal(result, 1.0)
    return result


def kendall_correlation(values: ArrayLike) -> NDArray[np.float64]:
    """Estimate elliptical rho from Kendall tau using sin(pi*tau/2)."""

    observations = np.asarray(values, dtype=float)
    if observations.ndim != 2 or observations.shape[0] < 2:
        raise ValueError("values must be a 2-D sample")
    n_assets = observations.shape[1]
    tau = np.eye(n_assets)
    for row in range(n_assets):
        for column in range(row + 1, n_assets):
            value = stats.kendalltau(
                observations[:, row], observations[:, column]
            ).statistic
            tau[row, column] = tau[column, row] = value
    rho = np.sin(np.pi * tau / 2.0)
    np.fill_diagonal(rho, 1.0)
    return fix_correlation(rho)


def multivariate_t_scale(
    standard_deviations: ArrayLike, correlation: ArrayLike, nu: float
) -> NDArray[np.float64]:
    """Convert desired standard deviations to a multivariate-t scale matrix."""

    if nu <= 2.0:
        raise ValueError("nu must exceed two for finite covariance")
    sd = np.asarray(standard_deviations, dtype=float).reshape(-1)
    rho = np.asarray(correlation, dtype=float)
    scaled = sd * np.sqrt((nu - 2.0) / nu)
    return np.outer(scaled, scaled) * rho


def multivariate_t_log_likelihood(
    values: ArrayLike,
    mean: ArrayLike,
    standard_deviations: ArrayLike,
    correlation: ArrayLike,
    nu: float,
) -> float:
    """Log likelihood under the course's profiled multivariate-t model."""

    observations = np.asarray(values, dtype=float)
    location = np.asarray(mean, dtype=float).reshape(-1)
    scale = multivariate_t_scale(standard_deviations, correlation, nu)
    return float(
        np.sum(stats.multivariate_t.logpdf(observations, loc=location, shape=scale, df=nu))
    )


def profile_nu(log_likelihood, *, low: float = 0.01, high: float = 0.49, n: int = 200):
    """Two-stage grid profile on theta=1/nu, matching the course reference."""

    theta = np.linspace(low, high, n)
    likelihoods = np.array([log_likelihood(1.0 / value) for value in theta])
    maximum = int(np.argmax(likelihoods))
    left = theta[max(maximum - 1, 0)]
    right = theta[min(maximum + 1, n - 1)]
    fine = np.linspace(left, right, n)
    fine_likelihoods = np.array([log_likelihood(1.0 / value) for value in fine])
    fine_maximum = int(np.argmax(fine_likelihoods))
    return float(1.0 / fine[fine_maximum]), float(fine_likelihoods[fine_maximum])


@dataclass(frozen=True)
class MultivariateTFit:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    nu: float
    log_likelihood: float
    correlation: NDArray[np.float64]
    standard_deviations: NDArray[np.float64]


def fit_multivariate_t(values: ArrayLike) -> MultivariateTFit:
    """Fit mean, Kendall correlation, scale, and profiled nu."""

    observations = np.asarray(values, dtype=float)
    mean = np.mean(observations, axis=0)
    standard_deviations = np.std(observations, axis=0, ddof=1)
    correlation = kendall_correlation(observations)
    nu, likelihood = profile_nu(
        lambda degrees: multivariate_t_log_likelihood(
            observations, mean, standard_deviations, correlation, degrees
        )
    )
    return MultivariateTFit(
        mean=mean,
        scale=multivariate_t_scale(standard_deviations, correlation, nu),
        nu=nu,
        log_likelihood=likelihood,
        correlation=correlation,
        standard_deviations=standard_deviations,
    )


def gaussian_copula_log_likelihood(
    uniforms: ArrayLike, correlation: ArrayLike
) -> float:
    """Gaussian copula log likelihood (joint minus marginal log densities)."""

    u = np.clip(np.asarray(uniforms, dtype=float), 1e-12, 1.0 - 1e-12)
    rho = np.asarray(correlation, dtype=float)
    transformed = stats.norm.ppf(u)
    joint = stats.multivariate_normal.logpdf(transformed, mean=np.zeros(rho.shape[0]), cov=rho)
    margins = stats.norm.logpdf(transformed).sum(axis=1)
    return float(np.sum(joint - margins))


def t_copula_log_likelihood(
    uniforms: ArrayLike, correlation: ArrayLike, nu: float
) -> float:
    """Student-t copula log likelihood (joint minus marginal log densities)."""

    u = np.clip(np.asarray(uniforms, dtype=float), 1e-12, 1.0 - 1e-12)
    rho = np.asarray(correlation, dtype=float)
    transformed = stats.t.ppf(u, df=nu)
    joint = stats.multivariate_t.logpdf(
        transformed, loc=np.zeros(rho.shape[0]), shape=rho, df=nu
    )
    margins = stats.t.logpdf(transformed, df=nu).sum(axis=1)
    return float(np.sum(joint - margins))


def fit_gaussian_copula(uniforms: ArrayLike):
    """Fit Kendall correlation and return Gaussian copula log likelihood."""

    correlation = kendall_correlation(uniforms)
    return correlation, gaussian_copula_log_likelihood(uniforms, correlation)


def fit_t_copula(uniforms: ArrayLike):
    """Fit Kendall correlation and profile the t-copula degrees of freedom."""

    correlation = kendall_correlation(uniforms)
    nu, likelihood = profile_nu(
        lambda degrees: t_copula_log_likelihood(uniforms, correlation, degrees)
    )
    return correlation, nu, likelihood


def simulate_gaussian_copula(
    correlation: ArrayLike, n_samples: int, *, seed: int = 1234
) -> NDArray[np.float64]:
    """Simulate Gaussian-copula uniforms through PCA Normal draws."""

    normal = simulate_pca(correlation, n_samples, seed=seed)
    return stats.norm.cdf(normal)


def simulate_t_copula(
    correlation: ArrayLike, nu: float, n_samples: int, *, seed: int = 1234
) -> NDArray[np.float64]:
    """Simulate t-copula uniforms with one common chi-square shock per row."""

    normal = simulate_pca(correlation, n_samples, seed=seed)
    rng = np.random.default_rng(seed + 1)
    scaling = np.sqrt(nu / rng.chisquare(nu, size=n_samples))
    return stats.t.cdf(normal * scaling[:, None], df=nu)


def tail_dependence_t(rho: float, nu: float) -> float:
    """Symmetric lower/upper tail-dependence coefficient of a t copula."""

    argument = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * stats.t.cdf(argument, df=nu + 1.0))


def copula_aicc(log_likelihood: float, parameters: int, observations: int) -> float:
    return float(
        -2.0 * log_likelihood
        + 2.0 * parameters
        + (2.0 * parameters**2 + 2.0 * parameters)
        / (observations - parameters - 1.0)
    )


def copula_bic(log_likelihood: float, parameters: int, observations: int) -> float:
    return float(parameters * np.log(observations) - 2.0 * log_likelihood)
