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
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .psd import higham_nearest_psd
from .risk_metrics import expected_shortfall, scipy_distribution, value_at_risk
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


def rank_uniforms(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to ranks scaled strictly inside the unit interval."""

    return frame.rank(method="average") / (len(frame) + 1.0)


def joint_tail_counts(
    uniforms: pd.DataFrame | NDArray[np.float64],
    *,
    threshold: float = 0.025,
    labels: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Count pairwise lower- and upper-tail co-occurrences."""

    values = np.asarray(uniforms, dtype=float)
    if values.ndim != 2:
        raise ValueError("uniforms must be a two-dimensional sample")
    names = list(labels) if labels is not None else [f"X{i + 1}" for i in range(values.shape[1])]
    if len(names) != values.shape[1]:
        raise ValueError("labels must match the number of columns")
    rows = []
    for left, right in combinations(range(values.shape[1]), 2):
        rows.append(
            {
                "Pair": f"{names[left]}-{names[right]}",
                "Worst Tail Count": int(
                    np.logical_and(
                        values[:, left] < threshold,
                        values[:, right] < threshold,
                    ).sum()
                ),
                "Best Tail Count": int(
                    np.logical_and(
                        values[:, left] > 1.0 - threshold,
                        values[:, right] > 1.0 - threshold,
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def expected_joint_tail_count(
    observations: int, *, threshold: float = 0.025
) -> float:
    """Return the expected pairwise same-tail count under independence."""

    return float(observations * threshold**2)


def scaled_joint_tail_counts(
    uniforms: pd.DataFrame | NDArray[np.float64],
    *,
    reference_observations: int,
    threshold: float = 0.025,
    labels: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Scale simulated pairwise tail counts to a reference sample size."""

    raw = joint_tail_counts(uniforms, threshold=threshold, labels=labels)
    simulated_observations = np.asarray(uniforms).shape[0]
    scale = reference_observations / simulated_observations
    raw["Worst Tail Count (scaled)"] = raw["Worst Tail Count"] * scale
    raw["Best Tail Count (scaled)"] = raw["Best Tail Count"] * scale
    return raw[
        ["Pair", "Worst Tail Count (scaled)", "Best Tail Count (scaled)"]
    ]


def tail_count_model_comparison(
    empirical: pd.DataFrame,
    gaussian: pd.DataFrame,
    student_t: pd.DataFrame,
) -> pd.DataFrame:
    """Compare observed pairwise tail counts with two fitted-copula implications."""

    observed = empirical.rename(
        columns={
            "Worst Tail Count": "Observed Worst",
            "Best Tail Count": "Observed Best",
        }
    )
    gaussian_counts = gaussian.rename(
        columns={
            "Worst Tail Count (scaled)": "Gaussian Worst",
            "Best Tail Count (scaled)": "Gaussian Best",
        }
    )
    student_counts = student_t.rename(
        columns={
            "Worst Tail Count (scaled)": "Student-t Worst",
            "Best Tail Count (scaled)": "Student-t Best",
        }
    )
    merged = observed.merge(gaussian_counts, on="Pair").merge(student_counts, on="Pair")
    rows = []
    for _, row in merged.iterrows():
        for tail in ("Worst", "Best"):
            rows.append(
                {
                    "Pair": row["Pair"],
                    "Tail": tail,
                    "Observed": row[f"Observed {tail}"],
                    "Gaussian": row[f"Gaussian {tail}"],
                    "Student-t": row[f"Student-t {tail}"],
                    "Gaussian Absolute Error": abs(
                        row[f"Gaussian {tail}"] - row[f"Observed {tail}"]
                    ),
                    "Student-t Absolute Error": abs(
                        row[f"Student-t {tail}"] - row[f"Observed {tail}"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def uniforms_from_fitted_margins(
    frame: pd.DataFrame, fitted: dict[str, object]
) -> pd.DataFrame:
    """Apply each fitted marginal CDF to its observed series."""

    missing = set(frame.columns) - set(fitted)
    if missing:
        raise ValueError(f"missing fitted margins for {sorted(missing)}")
    return pd.DataFrame(
        {
            column: scipy_distribution(fitted[column]).cdf(
                frame[column].to_numpy(dtype=float)
            )
            for column in frame
        },
        index=frame.index,
    ).clip(1e-12, 1.0 - 1e-12)


def fit_copula_models(
    uniforms: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit Gaussian and Student-t copulas and report likelihood/AICc/BIC."""

    values = uniforms.to_numpy(dtype=float)
    gaussian_correlation, gaussian_ll = fit_gaussian_copula(values)
    t_correlation, t_nu, t_ll = fit_t_copula(values)
    correlation_parameters = values.shape[1] * (values.shape[1] - 1) // 2
    gaussian_parameters = correlation_parameters
    t_parameters = correlation_parameters + 1
    table = pd.DataFrame(
        [
            {
                "Copula": "Gaussian",
                "Log Likelihood": gaussian_ll,
                "nu": np.nan,
                "AICc": copula_aicc(gaussian_ll, gaussian_parameters, len(values)),
                "BIC": copula_bic(gaussian_ll, gaussian_parameters, len(values)),
            },
            {
                "Copula": "Student-t",
                "Log Likelihood": t_ll,
                "nu": t_nu,
                "AICc": copula_aicc(t_ll, t_parameters, len(values)),
                "BIC": copula_bic(t_ll, t_parameters, len(values)),
            },
        ]
    )
    return table, {
        "gaussian_correlation": gaussian_correlation,
        "gaussian_log_likelihood": gaussian_ll,
        "t_correlation": t_correlation,
        "t_nu": t_nu,
        "t_log_likelihood": t_ll,
    }


def simulate_copula_returns(
    fitted_copulas: dict[str, object],
    fitted_margins: dict[str, object],
    columns: list[str] | tuple[str, ...],
    *,
    n_samples: int,
    seed: int = 545,
) -> dict[str, object]:
    """Simulate two fitted copulas and invert the selected marginal CDFs."""

    gaussian_uniforms = simulate_gaussian_copula(
        fitted_copulas["gaussian_correlation"], n_samples, seed=seed
    )
    t_uniforms = simulate_t_copula(
        fitted_copulas["t_correlation"],
        float(fitted_copulas["t_nu"]),
        n_samples,
        seed=seed,
    )

    def invert(uniform_values: NDArray[np.float64]) -> NDArray[np.float64]:
        simulated = np.empty_like(uniform_values)
        for index, column in enumerate(columns):
            simulated[:, index] = scipy_distribution(fitted_margins[column]).ppf(
                uniform_values[:, index]
            )
        return simulated

    return {
        "gaussian_uniforms": gaussian_uniforms,
        "gaussian_returns": invert(gaussian_uniforms),
        "t_uniforms": t_uniforms,
        "t_returns": invert(t_uniforms),
    }


def portfolio_tail_risk_comparison(
    historical_returns: pd.DataFrame,
    simulations: dict[str, object],
    *,
    position_values: ArrayLike,
    alphas: tuple[float, ...] = (0.05, 0.01),
) -> pd.DataFrame:
    """Compare historical and simulated linear-portfolio VaR and ES."""

    values = np.asarray(position_values, dtype=float).reshape(-1)
    sources = {
        "Historical": historical_returns.to_numpy(dtype=float),
        "Gaussian copula": np.asarray(simulations["gaussian_returns"], dtype=float),
        "Student-t copula": np.asarray(simulations["t_returns"], dtype=float),
    }
    rows = []
    for source, returns in sources.items():
        pnl = returns @ values
        for alpha in alphas:
            rows.append(
                {
                    "Source": source,
                    "Alpha": alpha,
                    "VaR ($)": value_at_risk(pnl, alpha),
                    "ES ($)": expected_shortfall(pnl, alpha),
                }
            )
    return pd.DataFrame(rows)


def copula_log_likelihood_contributions(
    uniforms: pd.DataFrame,
    fitted_copulas: dict[str, object],
) -> pd.DataFrame:
    """Return observation-level Student-t-minus-Gaussian copula log likelihood."""

    values = np.clip(uniforms.to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
    gaussian_correlation = np.asarray(
        fitted_copulas["gaussian_correlation"], dtype=float
    )
    t_correlation = np.asarray(fitted_copulas["t_correlation"], dtype=float)
    nu = float(fitted_copulas["t_nu"])

    normal_scores = stats.norm.ppf(values)
    gaussian_rows = stats.multivariate_normal.logpdf(
        normal_scores,
        mean=np.zeros(values.shape[1]),
        cov=gaussian_correlation,
    ) - stats.norm.logpdf(normal_scores).sum(axis=1)

    t_scores = stats.t.ppf(values, df=nu)
    t_rows = stats.multivariate_t.logpdf(
        t_scores,
        loc=np.zeros(values.shape[1]),
        shape=t_correlation,
        df=nu,
    ) - stats.t.logpdf(t_scores, df=nu).sum(axis=1)

    no_outer_five = np.logical_and(values >= 0.05, values <= 0.95).all(axis=1)
    return pd.DataFrame(
        {
            "t minus Gaussian": t_rows - gaussian_rows,
            "No series in outer 5%": no_outer_five,
        },
        index=uniforms.index,
    )


def summarize_log_likelihood_contributions(
    contributions: pd.DataFrame,
) -> dict[str, float]:
    """Summarize whole-sample and central-region copula likelihood gains."""

    likelihood_difference = contributions["t minus Gaussian"]
    middle_mask = contributions["No series in outer 5%"]
    total = float(likelihood_difference.sum())
    middle = float(likelihood_difference[middle_mask].sum())
    outer = total - middle
    middle_count = int(middle_mask.sum())
    outer_count = int((~middle_mask).sum())
    return {
        "Total t-minus-Gaussian Log Likelihood": total,
        "Contribution With No Series in Outer 5%": middle,
        "Fraction From Non-Outer-5% Days": middle / total,
        "Fraction of Days With No Series in Outer 5%": middle_count
        / len(contributions),
        "Average Contribution per Non-Outer-5% Day": middle / middle_count,
        "Average Contribution per Outer-5% Day": outer / outer_count,
        "Outer-to-Non-Outer Average Contribution Ratio": (outer / outer_count)
        / (middle / middle_count),
    }


def copula_risk_differences(risk_table: pd.DataFrame) -> pd.DataFrame:
    """Compare Student-t- and Gaussian-copula portfolio risk estimates."""

    simulated = risk_table[
        risk_table["Source"].isin(["Gaussian copula", "Student-t copula"])
    ]
    pivot = simulated.pivot(index="Alpha", columns="Source", values=["VaR ($)", "ES ($)"])
    rows = []
    for alpha in sorted(pivot.index, reverse=True):
        rows.append(
            {
                "Alpha": alpha,
                "t minus Gaussian VaR ($)": (
                    pivot.loc[alpha, ("VaR ($)", "Student-t copula")]
                    - pivot.loc[alpha, ("VaR ($)", "Gaussian copula")]
                ),
                "t minus Gaussian ES ($)": (
                    pivot.loc[alpha, ("ES ($)", "Student-t copula")]
                    - pivot.loc[alpha, ("ES ($)", "Gaussian copula")]
                ),
            }
        )
    return pd.DataFrame(rows)


def strongest_pair_tail_dependence(
    fitted_copulas: dict[str, object], labels: list[str] | tuple[str, ...]
) -> dict[str, float | str]:
    """Find the strongest fitted t-copula pair and calculate its tail dependence."""

    correlation = np.asarray(fitted_copulas["t_correlation"], dtype=float)
    candidates = [
        (correlation[left, right], left, right)
        for left, right in combinations(range(correlation.shape[0]), 2)
    ]
    rho, left, right = max(candidates)
    return {
        "Pair": f"{labels[left]}-{labels[right]}",
        "rho": float(rho),
        "Student-t Tail Dependence": tail_dependence_t(
            float(rho), float(fitted_copulas["t_nu"])
        ),
        "Gaussian Tail Dependence": 0.0,
    }
