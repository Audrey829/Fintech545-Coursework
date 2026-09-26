"""Ex-post return and volatility attribution for Functional Tests 11.1-11.2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ExPostAttribution:
    """Full attribution result, including the course output table."""

    attribution: pd.DataFrame
    weights: NDArray[np.float64]
    factor_weights: NDArray[np.float64]
    residual_individual: NDArray[np.float64]
    residual_return: NDArray[np.float64]
    portfolio_return: NDArray[np.float64]
    carino_k: NDArray[np.float64]


def _compound_return(values: NDArray[np.float64]) -> float:
    return float(np.exp(np.log1p(values).sum()) - 1.0)


def expost_factor(
    starting_weights: ArrayLike,
    stock_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    betas: ArrayLike,
) -> ExPostAttribution:
    """Calculate Carino-linked return attribution and realized-volatility attribution.

    This is the calculation used in Tests 11.1 and 11.2. Dynamic portfolio
    weights are updated from each period's stock returns before the next row.
    """

    weights0 = np.asarray(starting_weights, dtype=float).reshape(-1)
    stock = stock_returns.to_numpy(dtype=float)
    factors = factor_returns.to_numpy(dtype=float)
    beta = np.asarray(betas, dtype=float)
    if stock.ndim != 2 or factors.ndim != 2 or stock.shape[0] != factors.shape[0]:
        raise ValueError("stock and factor returns must have the same row count")
    if stock.shape[1] != weights0.size or beta.shape != (stock.shape[1], factors.shape[1]):
        raise ValueError("weights/betas dimensions do not match the return columns")
    if not np.isclose(weights0.sum(), 1.0):
        raise ValueError("starting weights must sum to one")

    n_observations = stock.shape[0]
    weights = np.empty((n_observations, weights0.size))
    factor_weights = np.empty((n_observations, factors.shape[1]))
    portfolio_return = np.empty(n_observations)
    residual_return = np.empty(n_observations)
    last_weights = weights0.copy()
    residual_individual = stock - factors @ beta.T

    for row in range(n_observations):
        weights[row] = last_weights
        factor_weights[row] = (beta * last_weights[:, None]).sum(axis=0)
        updated = last_weights * (1.0 + stock[row])
        gross_portfolio = float(updated.sum())
        last_weights = updated / gross_portfolio
        portfolio_return[row] = gross_portfolio - 1.0
        residual_return[row] = (
            portfolio_return[row] - factor_weights[row] @ factors[row]
        )

    total_return = _compound_return(portfolio_return)
    overall_k = np.log1p(total_return) / total_return
    period_ratio = np.divide(
        np.log1p(portfolio_return),
        portfolio_return,
        out=np.ones_like(portfolio_return),
        where=portfolio_return != 0.0,
    )
    carino_k = period_ratio / overall_k
    attributed = factors * factor_weights * carino_k[:, None]
    alpha_attributed = residual_return * carino_k
    residual_individual = residual_individual * weights

    factor_names = list(factor_returns.columns)
    columns = factor_names + ["Alpha", "Portfolio"]
    total_row = [_compound_return(factors[:, i]) for i in range(factors.shape[1])]
    total_row.extend([_compound_return(residual_return), total_return])
    attribution_row = attributed.sum(axis=0).tolist()
    attribution_row.extend([float(alpha_attributed.sum()), total_return])

    weighted_components = np.column_stack(
        [factors * factor_weights, residual_return]
    )
    design = np.column_stack([np.ones(n_observations), portfolio_return])
    regression = np.linalg.solve(design.T @ design, design.T @ weighted_components)
    component_sd = regression[1] * np.std(portfolio_return, ddof=1)
    volatility_row = component_sd.tolist() + [float(np.std(portfolio_return, ddof=1))]

    table = pd.DataFrame(
        [total_row, attribution_row, volatility_row],
        columns=columns,
    )
    table.insert(0, "Value", ["TotalReturn", "Return Attribution", "Vol Attribution"])
    return ExPostAttribution(
        attribution=table,
        weights=weights,
        factor_weights=factor_weights,
        residual_individual=residual_individual,
        residual_return=residual_return,
        portfolio_return=portfolio_return,
        carino_k=carino_k,
    )
