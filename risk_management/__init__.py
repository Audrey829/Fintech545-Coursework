"""Reusable quantitative-risk functions for FinTech 545.

The public API intentionally follows the names used in the course functional
tests while using normal Python naming conventions.

Functional-test module map
--------------------------
- Tests 1.1-2.3: ``covariance.py``
- Tests 3.1-4.1: ``psd.py``
- Tests 5.1-5.5: ``simulation.py``
- Tests 6.1-6.2: ``returns.py``
- Tests 7.1-7.6: ``distributions.py``
"""

from .covariance import (
    covariance_with_ew_variance,
    ew_covariance,
    ew_weights,
    missing_covariance,
)
from .distributions import (
    NormalFit,
    NormalInverseGaussianFit,
    StudentTFit,
    TRegressionFit,
    aicc,
    fit_nig_mle,
    fit_nig_moments,
    fit_normal,
    fit_student_t,
    fit_t_regression,
)
from .psd import chol_psd, higham_nearest_psd, near_psd
from .returns import calculate_returns
from .simulation import simulate_normal, simulate_pca

# Course-style wrappers make the package easy to use beside the Julia reference.
def missing_cov(values, *, skipMiss=True, fun="cov"):
    """Julia-compatible wrapper for :func:`missing_covariance`."""

    function_name = fun if isinstance(fun, str) else getattr(fun, "__name__", "")
    correlation = function_name.lower() in {"cor", "corr", "corrcoef", "correlation"}
    return missing_covariance(
        values, skip_missing=skipMiss, correlation=correlation
    )


def ewCovar(values, decay):
    """Julia-compatible wrapper for :func:`ew_covariance`."""

    return ew_covariance(values, decay)


def higham_nearestPSD(
    matrix, weight=None, epsilon=1e-9, maxIter=100, tol=1e-9
):
    """Julia-compatible wrapper for :func:`higham_nearest_psd`."""

    return higham_nearest_psd(
        matrix,
        weight,
        epsilon=epsilon,
        max_iterations=maxIter,
        tolerance=tol,
    )


def simulateNormal(
    n_samples, covariance, *, mean=None, seed=1234, fixMethod=near_psd
):
    """Julia-compatible wrapper for :func:`simulate_normal`."""

    return simulate_normal(
        n_samples,
        covariance,
        mean=mean,
        seed=seed,
        fix_method=fixMethod,
    )


def return_calculate(prices, *, method="DISCRETE", dateColumn="date"):
    """Julia-compatible wrapper for :func:`calculate_returns`."""

    return calculate_returns(prices, method=method, date_column=dateColumn)


fit_general_t = fit_student_t
fit_regression_t = fit_t_regression
fit_NIG_mle = fit_nig_mle

__all__ = [
    "NormalFit",
    "NormalInverseGaussianFit",
    "StudentTFit",
    "TRegressionFit",
    "aicc",
    "calculate_returns",
    "chol_psd",
    "covariance_with_ew_variance",
    "ew_covariance",
    "ew_weights",
    "fit_normal",
    "fit_nig_mle",
    "fit_nig_moments",
    "fit_NIG_mle",
    "fit_student_t",
    "fit_t_regression",
    "fit_general_t",
    "fit_regression_t",
    "higham_nearest_psd",
    "higham_nearestPSD",
    "missing_cov",
    "missing_covariance",
    "near_psd",
    "return_calculate",
    "simulateNormal",
    "simulate_normal",
    "simulate_pca",
    "ewCovar",
]
