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

from .distributions import NormalFit, NormalInverseGaussianFit, StudentTFit


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
