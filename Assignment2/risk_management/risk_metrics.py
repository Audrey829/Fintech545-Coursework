"""Value-at-Risk and Expected Shortfall for Functional Tests 8, 9, and 13.9.

Test/function map
-----------------
- Tests 8.1-8.6: distribution and empirical ``value_at_risk`` /
  ``expected_shortfall``
- Test 9.1: ``aggregate_portfolio_risk`` after Gaussian-copula simulation
- Test 13.9: ``aggregate_portfolio_risk`` after t-copula simulation

Every function in this module calculates from values or fitted parameters
passed by the caller. Professor ``testout`` files are never read here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, stats

from .covariance import ew_covariance, ew_weights, portfolio_variance
from .distributions import (
    NormalFit,
    NormalInverseGaussianFit,
    StudentTFit,
    fit_normal,
    fit_student_t,
)


def _finite_vector(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("values must contain finite observations")
    return result


def scipy_distribution(model):
    """Convert a course fit object to the corresponding frozen SciPy model."""

    if isinstance(model, NormalFit):
        return stats.norm(loc=model.mu, scale=model.sigma)
    if isinstance(model, StudentTFit):
        return stats.t(df=model.nu, loc=model.mu, scale=model.sigma)
    if isinstance(model, NormalInverseGaussianFit):
        a, b, loc, scale = model.scipy_parameters
        return stats.norminvgauss(a, b, loc=loc, scale=scale)
    if hasattr(model, "ppf") and hasattr(model, "pdf"):
        return model
    raise TypeError("model must be a supported fitted or frozen distribution")


def value_at_risk(values_or_model, alpha: float = 0.05) -> float:
    """Return positive loss VaR using the course's empirical convention.

    For samples, the floor and ceiling order statistics around ``n * alpha``
    are averaged, matching ``RiskStats.jl``. For distributions, the result is
    the negative lower-tail quantile.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if isinstance(values_or_model, (NormalFit, StudentTFit, NormalInverseGaussianFit)) \
            or (hasattr(values_or_model, "ppf") and hasattr(values_or_model, "pdf")):
        return float(-scipy_distribution(values_or_model).ppf(alpha))

    values = np.sort(_finite_vector(values_or_model))
    n = values.size
    upper = max(0, int(np.ceil(n * alpha)) - 1)
    lower = max(0, int(np.floor(n * alpha)) - 1)
    return float(-0.5 * (values[upper] + values[lower]))


def expected_shortfall(values_or_model, alpha: float = 0.05) -> float:
    """Return positive loss Expected Shortfall below the ``alpha`` quantile."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if isinstance(values_or_model, (NormalFit, StudentTFit, NormalInverseGaussianFit)) \
            or (hasattr(values_or_model, "ppf") and hasattr(values_or_model, "pdf")):
        distribution = scipy_distribution(values_or_model)
        lower = float(distribution.ppf(1e-12))
        cutoff = float(distribution.ppf(alpha))
        integral = integrate.quad(
            lambda x: x * distribution.pdf(x),
            lower,
            cutoff,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=250,
        )[0]
        return float(-integral / alpha)

    values = np.sort(_finite_vector(values_or_model))
    cutoff = -value_at_risk(values, alpha)
    return float(-np.mean(values[values <= cutoff]))


def simulate_fitted_distribution(
    model,
    n_samples: int,
    *,
    seed: int = 8,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Simulate a fitted univariate model by inverse-CDF transformation."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    generator = np.random.default_rng(seed) if rng is None else rng
    uniforms = generator.random(n_samples)
    return np.asarray(scipy_distribution(model).ppf(uniforms), dtype=float)


def aggregate_portfolio_risk(
    simulated_returns: ArrayLike,
    current_values: ArrayLike,
    asset_names: list[str] | tuple[str, ...],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Calculate asset and total portfolio VaR/ES from simulated returns."""

    returns = np.asarray(simulated_returns, dtype=float)
    values = _finite_vector(current_values)
    if returns.ndim != 2 or returns.shape[1] != values.size:
        raise ValueError("simulated_returns must have one column per current value")
    if len(asset_names) != values.size:
        raise ValueError("asset_names must have one name per current value")
    if not np.isfinite(returns).all() or np.any(values <= 0.0):
        raise ValueError("returns must be finite and current values positive")

    pnl = returns * values
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(asset_names):
        var = value_at_risk(pnl[:, index], alpha)
        es = expected_shortfall(pnl[:, index], alpha)
        rows.append(
            {
                "Stock": str(name),
                "VaR95": var,
                "ES95": es,
                "VaR95_Pct": var / values[index],
                "ES95_Pct": es / values[index],
            }
        )

    total_pnl = pnl.sum(axis=1)
    total_value = float(values.sum())
    total_var = value_at_risk(total_pnl, alpha)
    total_es = expected_shortfall(total_pnl, alpha)
    rows.append(
        {
            "Stock": "Total",
            "VaR95": total_var,
            "ES95": total_es,
            "VaR95_Pct": total_var / total_value,
            "ES95_Pct": total_es / total_value,
        }
    )
    return pd.DataFrame(rows)


def ewma_diagnostics(
    values: ArrayLike, decay: float, *, recent_days: int
) -> dict[str, float]:
    """Return EW volatility, effective sample size, half-life, and recent weight."""

    sample = _finite_vector(values)
    if not 1 <= recent_days <= sample.size:
        raise ValueError("recent_days must be between one and the sample size")
    weights = ew_weights(sample.size, decay)
    variance = ew_covariance(sample[:, None], decay)[0, 0]
    return {
        "Decay": float(decay),
        "Volatility": float(np.sqrt(variance)),
        "Effective Sample Size": float(1.0 / np.sum(weights**2)),
        "Half Life (days)": float(np.log(0.5) / np.log(decay)),
        f"Weight on Last {recent_days} Days": float(weights[-recent_days:].sum()),
    }


def var_model_comparison(
    centered_returns: ArrayLike,
    *,
    position_value: float,
    decays: tuple[float, ...] = (0.97, 0.94),
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, StudentTFit]:
    """Compare Normal, EW-Normal, Student-t, and historical one-day VaR."""

    sample = _finite_vector(centered_returns)
    normal_fit = fit_normal(sample)
    t_fit = fit_student_t(sample)
    rows = [
        {
            "Model": "Normal - equal weight",
            "VaR ($)": position_value * value_at_risk(normal_fit, alpha),
        }
    ]
    for decay in decays:
        volatility = np.sqrt(ew_covariance(sample[:, None], decay)[0, 0])
        rows.append(
            {
                "Model": f"Normal - EWMA lambda={decay:.2f}",
                "VaR ($)": position_value
                * value_at_risk(NormalFit(mu=0.0, sigma=float(volatility)), alpha),
            }
        )
    rows.extend(
        [
            {
                "Model": "Student-t MLE",
                "VaR ($)": position_value * value_at_risk(t_fit, alpha),
            },
            {
                "Model": "Historical simulation",
                "VaR ($)": position_value * value_at_risk(sample, alpha),
            },
        ]
    )
    return pd.DataFrame(rows).sort_values("VaR ($)").reset_index(drop=True), t_fit


def var_sampling_noise(
    volatility: float,
    effective_sample_size: float,
    *,
    position_value: float,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Approximate EW volatility and Normal-VaR sampling noise."""

    volatility_standard_error = volatility / np.sqrt(2.0 * effective_sample_size)
    return {
        "Volatility Standard Error": float(volatility_standard_error),
        "Dollar Volatility Standard Error": float(
            position_value * volatility_standard_error
        ),
        "Dollar VaR Standard Error": float(
            position_value
            * value_at_risk(
                NormalFit(mu=0.0, sigma=float(volatility_standard_error)), alpha
            )
        ),
    }


def pnl_from_return_scenarios(
    returns: pd.DataFrame, positions: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Build scenario P&L columns for named linear positions."""

    output: dict[str, NDArray[np.float64]] = {}
    for portfolio, holdings in positions.items():
        missing = set(holdings) - set(returns.columns)
        if missing:
            raise ValueError(f"unknown return columns: {sorted(missing)}")
        pnl = np.zeros(len(returns), dtype=float)
        for asset, notional in holdings.items():
            pnl += returns[asset].to_numpy(dtype=float) * float(notional)
        output[portfolio] = pnl
    return pd.DataFrame(output, index=returns.index)


def empirical_risk_table(
    pnl: pd.DataFrame, *, alpha: float = 0.05
) -> pd.DataFrame:
    """Compute empirical VaR and ES for each P&L column."""

    return pd.DataFrame(
        [
            {
                "Position": column,
                "VaR ($)": value_at_risk(pnl[column], alpha),
                "ES ($)": expected_shortfall(pnl[column], alpha),
            }
            for column in pnl
        ]
    )


def normal_risk_table(pnl: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Compute fitted-Normal VaR for each P&L column."""

    return pd.DataFrame(
        [
            {
                "Position": column,
                "Normal VaR ($)": value_at_risk(fit_normal(pnl[column]), alpha),
            }
            for column in pnl
        ]
    )


def threshold_event_counts(
    values: pd.DataFrame, *, threshold: float, below: bool = True
) -> dict[str, int]:
    """Count column, union, and intersection threshold events."""

    numeric = values.to_numpy(dtype=float)
    events = numeric < threshold if below else numeric > threshold
    result = {
        str(column): int(events[:, index].sum())
        for index, column in enumerate(values.columns)
    }
    result["At least one"] = int(events.any(axis=1).sum())
    result["All"] = int(events.all(axis=1).sum())
    return result


def subadditivity_check(
    risk_table: pd.DataFrame,
    *,
    component_a: str,
    component_b: str,
    combined: str,
) -> pd.DataFrame:
    """Check whether VaR and ES of a combination exceed component sums."""

    indexed = risk_table.set_index("Position")
    rows = []
    for metric in ("VaR ($)", "ES ($)"):
        component_sum = indexed.loc[component_a, metric] + indexed.loc[component_b, metric]
        combined_value = indexed.loc[combined, metric]
        rows.append(
            {
                "Risk Measure": metric.replace(" ($)", ""),
                "A + B standalone ($)": component_sum,
                "Combined ($)": combined_value,
                "Subadditive": bool(combined_value <= component_sum),
            }
        )
    return pd.DataFrame(rows)


def linear_portfolio_var(
    returns: ArrayLike,
    positions: dict[str, ArrayLike],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute empirical VaR for named linear portfolios."""

    simulated = np.asarray(returns, dtype=float)
    if simulated.ndim != 2:
        raise ValueError("returns must be a two-dimensional array")
    return pd.DataFrame(
        [
            {
                "Portfolio": name,
                "VaR ($)": value_at_risk(
                    simulated @ np.asarray(exposures, dtype=float).reshape(-1), alpha
                ),
            }
            for name, exposures in positions.items()
        ]
    )


def delta_normal_var(
    covariance: ArrayLike,
    positions: dict[str, ArrayLike],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute zero-mean delta-Normal VaR for named linear portfolios."""

    covariance_matrix = np.asarray(covariance, dtype=float)
    rows = []
    for name, exposures in positions.items():
        weights = np.asarray(exposures, dtype=float).reshape(-1)
        sigma = np.sqrt(portfolio_variance(covariance_matrix, weights))
        rows.append(
            {
                "Portfolio": name,
                "Delta-normal VaR ($)": value_at_risk(
                    NormalFit(mu=0.0, sigma=float(sigma)), alpha
                ),
            }
        )
    return pd.DataFrame(rows)
