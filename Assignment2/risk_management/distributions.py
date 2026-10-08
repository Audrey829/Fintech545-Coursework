"""Functions for Functional Tests 7.1 through 7.6.

Test/function map
-----------------
- Test 7.1: ``fit_normal``
- Test 7.2: ``fit_student_t``
- Test 7.3: ``fit_t_regression``
- Test 7.4: ``aicc``
- Test 7.5: ``fit_nig_moments``
- Test 7.6: ``fit_nig_mle``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats


def _sample(values: ArrayLike) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=float).reshape(-1)
    if out.size < 2 or not np.isfinite(out).all():
        raise ValueError("data must contain at least two finite observations")
    return out


@dataclass(frozen=True)
class NormalFit:
    mu: float
    sigma: float

    @property
    def parameter_count(self) -> int:
        return 2

    def log_likelihood(self, data: ArrayLike) -> float:
        return float(stats.norm.logpdf(_sample(data), loc=self.mu, scale=self.sigma).sum())


@dataclass(frozen=True)
class StudentTFit:
    mu: float
    sigma: float
    nu: float

    @property
    def parameter_count(self) -> int:
        return 3

    def log_likelihood(self, data: ArrayLike) -> float:
        return float(
            stats.t.logpdf(_sample(data), df=self.nu, loc=self.mu, scale=self.sigma).sum()
        )


@dataclass(frozen=True)
class NormalInverseGaussianFit:
    """Normal-inverse-Gaussian fit in the course parameterization."""

    mu: float
    alpha: float
    beta: float
    delta: float

    def __post_init__(self) -> None:
        if self.alpha <= abs(self.beta):
            raise ValueError("NIG parameters require alpha > abs(beta)")
        if self.delta <= 0.0:
            raise ValueError("NIG delta must be positive")

    @property
    def parameter_count(self) -> int:
        return 4

    @property
    def gamma(self) -> float:
        return float(np.sqrt(self.alpha**2 - self.beta**2))

    @property
    def mean(self) -> float:
        return float(self.mu + self.delta * self.beta / self.gamma)

    @property
    def variance(self) -> float:
        return float(self.delta * self.alpha**2 / self.gamma**3)

    @property
    def skewness(self) -> float:
        return float(
            3.0 * self.beta / (self.alpha * np.sqrt(self.delta * self.gamma))
        )

    @property
    def excess_kurtosis(self) -> float:
        return float(
            3.0
            * (1.0 + 4.0 * self.beta**2 / self.alpha**2)
            / (self.delta * self.gamma)
        )

    @property
    def scipy_parameters(self) -> tuple[float, float, float, float]:
        """Return SciPy's ``(a, b, loc, scale)`` parameterization."""

        return (
            self.alpha * self.delta,
            self.beta * self.delta,
            self.mu,
            self.delta,
        )

    def log_likelihood(self, data: ArrayLike) -> float:
        a, b, loc, scale = self.scipy_parameters
        return float(
            stats.norminvgauss.logpdf(
                _sample(data), a, b, loc=loc, scale=scale
            ).sum()
        )


@dataclass(frozen=True)
class TRegressionFit:
    alpha: float
    beta: NDArray[np.float64]
    sigma: float
    nu: float
    residuals: NDArray[np.float64]

    @property
    def mu(self) -> float:
        return 0.0

    @property
    def parameter_count(self) -> int:
        return 3 + 1 + self.beta.size

    def predict(self, predictors: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(predictors, dtype=float)
        return self.alpha + x @ self.beta

    def log_likelihood(self, _: ArrayLike | None = None) -> float:
        return float(stats.t.logpdf(self.residuals, self.nu, scale=self.sigma).sum())


def fit_normal(data: ArrayLike) -> NormalFit:
    """Fit a Normal distribution using mean and sample standard deviation."""

    x = _sample(data)
    return NormalFit(mu=float(np.mean(x)), sigma=float(np.std(x, ddof=1)))


def fit_student_t(data: ArrayLike) -> StudentTFit:
    """Maximum-likelihood location-scale Student-t fit."""

    x = _sample(data)
    nu, mu, sigma = stats.t.fit(x)
    return StudentTFit(mu=float(mu), sigma=float(sigma), nu=float(nu))


def fit_nig_moments(data: ArrayLike) -> NormalInverseGaussianFit:
    """Fit an NIG distribution by matching four sample moments.

    The sample variance uses ``ddof=1`` while skewness and excess kurtosis use
    their ordinary moment estimators. This matches the course reference.
    """

    x = _sample(data)
    sample_mean = float(np.mean(x))
    sample_variance = float(np.var(x, ddof=1))
    sample_skewness = float(stats.skew(x, bias=True))
    sample_excess_kurtosis = float(stats.kurtosis(x, fisher=True, bias=True))

    if sample_variance <= 0.0:
        raise ValueError("NIG method of moments needs positive sample variance")
    if sample_excess_kurtosis <= 0.0:
        raise ValueError(
            "NIG method of moments needs positive excess kurtosis, "
            f"got {sample_excess_kurtosis}"
        )

    shape_ratio = sample_skewness**2 / sample_excess_kurtosis
    if shape_ratio >= 3.0 / 5.0:
        raise ValueError(
            "sample is outside the NIG moment region: excess kurtosis must "
            "exceed (5/3) * skewness^2"
        )

    rho_squared = shape_ratio / (3.0 - 4.0 * shape_ratio)
    rho = float(np.sign(sample_skewness) * np.sqrt(rho_squared))
    delta_gamma = (
        3.0 * (1.0 + 4.0 * rho_squared) / sample_excess_kurtosis
    )
    alpha = float(
        np.sqrt(
            delta_gamma
            / (sample_variance * (1.0 - rho_squared) ** 2)
        )
    )
    beta = rho * alpha
    gamma = alpha * np.sqrt(1.0 - rho_squared)
    delta = float(delta_gamma / gamma)
    mu = float(sample_mean - delta * beta / gamma)

    return NormalInverseGaussianFit(mu, alpha, beta, delta)


def fit_nig_mle(data: ArrayLike) -> NormalInverseGaussianFit:
    """Fit an NIG distribution by maximum likelihood using SciPy."""

    x = _sample(data)
    a, b, mu, delta = stats.norminvgauss.fit(x)
    alpha = float(a / delta)
    beta = float(b / delta)
    return NormalInverseGaussianFit(
        mu=float(mu), alpha=alpha, beta=beta, delta=float(delta)
    )


def fit_t_regression(y: ArrayLike, predictors: ArrayLike) -> TRegressionFit:
    """Maximum-likelihood linear regression with centered Student-t errors."""

    response = _sample(y)
    x = np.asarray(predictors, dtype=float)
    if x.ndim != 2 or x.shape[0] != response.size:
        raise ValueError("predictors must be a 2-D array with one row per response")
    design = np.column_stack([np.ones(response.size), x])
    start_beta = np.linalg.lstsq(design, response, rcond=None)[0]
    start_residuals = response - design @ start_beta
    excess_kurtosis = stats.kurtosis(start_residuals, fisher=True, bias=False)
    start_nu = max(2.0001, 6.0 / excess_kurtosis + 4.0) if excess_kurtosis > 0 else 30.0
    start_sigma = np.std(start_residuals, ddof=1) * np.sqrt(
        (start_nu - 2.0) / start_nu
    )

    initial = np.concatenate(
        [start_beta, [np.log(max(start_sigma, 1e-6)), np.log(start_nu - 2.0)]]
    )
    n_beta = design.shape[1]

    def negative_log_likelihood(parameters: NDArray[np.float64]) -> float:
        beta = parameters[:n_beta]
        sigma = np.exp(parameters[n_beta])
        nu = 2.0 + np.exp(parameters[n_beta + 1])
        residuals = response - design @ beta
        return float(-stats.t.logpdf(residuals, df=nu, loc=0.0, scale=sigma).sum())

    fitted = optimize.minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        options={"maxiter": 10_000, "ftol": 1e-13, "gtol": 1e-10},
    )
    if not fitted.success:
        raise RuntimeError(f"Student-t regression failed: {fitted.message}")

    beta = fitted.x[:n_beta]
    sigma = float(np.exp(fitted.x[n_beta]))
    nu = float(2.0 + np.exp(fitted.x[n_beta + 1]))
    residuals = response - design @ beta
    return TRegressionFit(
        alpha=float(beta[0]),
        beta=np.asarray(beta[1:], dtype=float),
        sigma=sigma,
        nu=nu,
        residuals=np.asarray(residuals, dtype=float),
    )


def aicc(
    model: NormalFit | StudentTFit | NormalInverseGaussianFit | TRegressionFit,
    data: ArrayLike,
) -> float:
    """Small-sample corrected Akaike Information Criterion."""

    n_observations = _sample(data).size
    k = model.parameter_count
    if n_observations <= k + 1:
        raise ValueError("AICc requires more than k + 1 observations")
    log_likelihood = model.log_likelihood(data)
    return float(
        -2.0 * log_likelihood
        + 2.0 * k
        + 2.0 * k * (k + 1.0) / (n_observations - k - 1.0)
    )


def standardized_tail_quantiles(
    student_degrees_of_freedom: float, *, alpha: float = 0.05
) -> pd.DataFrame:
    """Compare unit-variance Normal and Student-t lower-tail quantiles."""

    nu = float(student_degrees_of_freedom)
    if nu <= 2.0:
        raise ValueError("Student-t degrees of freedom must exceed two")
    return pd.DataFrame(
        [
            {
                "Distribution": "Normal",
                "Lower-tail quantile": stats.norm.ppf(alpha),
            },
            {
                "Distribution": f"Student-t (nu={nu:.4f})",
                "Lower-tail quantile": stats.t.ppf(alpha, df=nu)
                * np.sqrt((nu - 2.0) / nu),
            },
        ]
    )


def fit_best_margins(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, NormalFit | StudentTFit]]:
    """Fit Normal and Student-t margins and select each column by AICc."""

    rows: list[dict[str, float | str]] = []
    chosen: dict[str, NormalFit | StudentTFit] = {}
    for column in frame:
        values = frame[column].to_numpy(dtype=float)
        normal = fit_normal(values)
        student = fit_student_t(values)
        normal_aicc = aicc(normal, values)
        student_aicc = aicc(student, values)
        winner = "Student-t" if student_aicc < normal_aicc else "Normal"
        chosen[column] = student if winner == "Student-t" else normal
        rows.append(
            {
                "Series": column,
                "Normal AICc": normal_aicc,
                "Student-t AICc": student_aicc,
                "Winner": winner,
                "Student-t nu": student.nu,
            }
        )
    return pd.DataFrame(rows), chosen
