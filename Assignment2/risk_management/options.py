"""Option pricing functions for Functional Tests 12.1 through 12.3.

Test/function map
-----------------
- Test 12.1: ``gbsm`` with analytic European-option Greeks
- Test 12.2: ``american_continuous`` and ``american_finite_difference_greeks``
- Test 12.3: ``american_discrete_dividends``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class OptionResult:
    value: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    carry_rho: float = 0.0


def gbsm(
    call: bool,
    underlying: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    cost_of_carry: float,
    implied_volatility: float,
    *,
    include_greeks: bool = False,
) -> OptionResult:
    """Generalized Black-Scholes-Merton value and analytic Greeks."""

    if underlying <= 0.0 or strike <= 0.0 or time_to_maturity <= 0.0:
        raise ValueError("underlying, strike, and maturity must be positive")
    if implied_volatility <= 0.0:
        raise ValueError("implied volatility must be positive")

    root_time = np.sqrt(time_to_maturity)
    d1 = (
        np.log(underlying / strike)
        + (cost_of_carry + implied_volatility**2 / 2.0) * time_to_maturity
    ) / (implied_volatility * root_time)
    d2 = d1 - implied_volatility * root_time
    carry_discount = np.exp((cost_of_carry - risk_free_rate) * time_to_maturity)
    strike_discount = np.exp(-risk_free_rate * time_to_maturity)

    if call:
        delta = carry_discount * stats.norm.cdf(d1)
        value = underlying * delta - strike * strike_discount * stats.norm.cdf(d2)
    else:
        delta = carry_discount * (stats.norm.cdf(d1) - 1.0)
        value = (
            strike * strike_discount * stats.norm.cdf(-d2)
            - underlying * carry_discount * stats.norm.cdf(-d1)
        )

    if not include_greeks:
        return OptionResult(value=float(value), delta=float(delta))

    density = stats.norm.pdf(d1)
    gamma = density * carry_discount / (
        underlying * implied_volatility * root_time
    )
    vega = underlying * carry_discount * density * root_time
    if call:
        theta = (
            -underlying * carry_discount * density * implied_volatility / (2.0 * root_time)
            - (cost_of_carry - risk_free_rate)
            * underlying
            * carry_discount
            * stats.norm.cdf(d1)
            - risk_free_rate * strike * strike_discount * stats.norm.cdf(d2)
        )
        rho = time_to_maturity * strike * strike_discount * stats.norm.cdf(d2)
        carry_rho = (
            time_to_maturity * underlying * carry_discount * stats.norm.cdf(d1)
        )
    else:
        theta = (
            -underlying * carry_discount * density * implied_volatility / (2.0 * root_time)
            + (cost_of_carry - risk_free_rate)
            * underlying
            * carry_discount
            * stats.norm.cdf(-d1)
            + risk_free_rate * strike * strike_discount * stats.norm.cdf(-d2)
        )
        rho = -time_to_maturity * strike * strike_discount * stats.norm.cdf(-d2)
        carry_rho = (
            -time_to_maturity * underlying * carry_discount * stats.norm.cdf(-d1)
        )
    return OptionResult(
        value=float(value),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
        carry_rho=float(carry_rho),
    )


def american_continuous(
    call: bool,
    underlying: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    cost_of_carry: float,
    implied_volatility: float,
    steps: int,
) -> float:
    """CRR binomial American option with a continuous cost of carry."""

    if steps < 1:
        raise ValueError("steps must be positive")
    dt = time_to_maturity / steps
    up = np.exp(implied_volatility * np.sqrt(dt))
    down = 1.0 / up
    up_probability = (np.exp(cost_of_carry * dt) - down) / (up - down)
    down_probability = 1.0 - up_probability
    discount = np.exp(-risk_free_rate * dt)
    sign = 1.0 if call else -1.0

    nodes = np.arange(steps + 1)
    prices = underlying * up**nodes * down ** (steps - nodes)
    option = np.maximum(0.0, sign * (prices - strike))
    for level in range(steps - 1, -1, -1):
        continuation = discount * (
            up_probability * option[1:] + down_probability * option[:-1]
        )
        nodes = np.arange(level + 1)
        prices = underlying * up**nodes * down ** (level - nodes)
        option = np.maximum(continuation, sign * (prices - strike))
    return float(option[0])


def american_finite_difference_greeks(
    call: bool,
    underlying: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    cost_of_carry: float,
    implied_volatility: float,
    steps: int = 500,
) -> OptionResult:
    """American value with central-difference Greeks used in Test 12.2.

    Rho differentiates the risk-free rate while holding cost of carry fixed,
    exactly as in the professor's six-parameter reference function. Theta is
    the derivative with respect to remaining maturity and is therefore the
    opposite sign of calendar-time decay.
    """

    parameters = np.array(
        [
            underlying,
            strike,
            time_to_maturity,
            risk_free_rate,
            cost_of_carry,
            implied_volatility,
        ],
        dtype=float,
    )

    def price(values: NDArray[np.float64]) -> float:
        return american_continuous(call, *values, steps)

    value = price(parameters)
    relative_step = np.cbrt(np.finfo(float).eps)
    gradient = np.empty(parameters.size)
    for index, parameter in enumerate(parameters):
        step = relative_step * max(1.0, abs(parameter))
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        gradient[index] = (price(upper) - price(lower)) / (2.0 * step)

    gamma_step = 1.5
    upper = parameters.copy()
    lower = parameters.copy()
    upper[0] += gamma_step
    lower[0] -= gamma_step
    gamma = (price(upper) + price(lower) - 2.0 * value) / gamma_step**2
    return OptionResult(
        value=value,
        delta=float(gradient[0]),
        gamma=float(gamma),
        vega=float(gradient[5]),
        rho=float(gradient[3]),
        theta=float(gradient[2]),
    )


def _american_no_dividend_batch(
    call: bool,
    underlyings: ArrayLike,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    implied_volatility: float,
    steps: int,
) -> NDArray[np.float64]:
    """Vectorized no-dividend CRR engine used inside discrete-dividend trees."""

    spots = np.asarray(underlyings, dtype=float).reshape(-1)
    if steps == 0:
        sign = 1.0 if call else -1.0
        return np.maximum(0.0, sign * (spots - strike))
    dt = time_to_maturity / steps
    up = np.exp(implied_volatility * np.sqrt(dt))
    down = 1.0 / up
    probability = (np.exp(risk_free_rate * dt) - down) / (up - down)
    discount = np.exp(-risk_free_rate * dt)
    sign = 1.0 if call else -1.0
    nodes = np.arange(steps + 1)
    prices = spots[:, None] * up**nodes * down ** (steps - nodes)
    option = np.maximum(0.0, sign * (prices - strike))
    for level in range(steps - 1, -1, -1):
        continuation = discount * (
            probability * option[:, 1:] + (1.0 - probability) * option[:, :-1]
        )
        nodes = np.arange(level + 1)
        prices = spots[:, None] * up**nodes * down ** (level - nodes)
        option = np.maximum(continuation, sign * (prices - strike))
    return option[:, 0]


def american_discrete_dividends(
    call: bool,
    underlying: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_amounts: ArrayLike,
    dividend_steps: ArrayLike,
    implied_volatility: float,
    steps: int,
) -> float:
    """CRR American option with fixed cash dividends at tree step numbers.

    The implementation evaluates the same recursive subtrees as the Julia
    reference but batches equal-length terminal subproblems. This preserves
    the model while avoiding tens of thousands of Python-level function calls.
    """

    amounts = np.asarray(dividend_amounts, dtype=float).reshape(-1)
    dates = np.asarray(dividend_steps, dtype=int).reshape(-1)
    if amounts.size != dates.size:
        raise ValueError("dividend amounts and steps must have the same length")
    active = (dates >= 0) & (dates <= steps)
    amounts = amounts[active]
    dates = dates[active]
    if amounts.size == 0:
        return american_continuous(
            call,
            underlying,
            strike,
            time_to_maturity,
            risk_free_rate,
            risk_free_rate,
            implied_volatility,
            steps,
        )
    order = np.argsort(dates)
    amounts = amounts[order]
    dates = dates[order]
    if amounts.size != 2:
        return _american_discrete_recursive(
            call,
            underlying,
            strike,
            time_to_maturity,
            risk_free_rate,
            amounts,
            dates,
            implied_volatility,
            steps,
        )

    dt = time_to_maturity / steps
    up = np.exp(implied_volatility * np.sqrt(dt))
    down = 1.0 / up
    probability = (np.exp(risk_free_rate * dt) - down) / (up - down)
    discount = np.exp(-risk_free_rate * dt)
    sign = 1.0 if call else -1.0
    first_steps = int(dates[0])
    middle_steps = int(dates[1] - dates[0])
    final_steps = int(steps - dates[1])

    first_nodes = np.arange(first_steps + 1)
    pre_first = underlying * up**first_nodes * down ** (first_steps - first_nodes)
    ex_first = pre_first - amounts[0]
    middle_nodes = np.arange(middle_steps + 1)
    pre_second = (
        ex_first[:, None]
        * up**middle_nodes
        * down ** (middle_steps - middle_nodes)
    )
    ex_second = pre_second - amounts[1]
    terminal = _american_no_dividend_batch(
        call,
        ex_second.reshape(-1),
        strike,
        final_steps * dt,
        risk_free_rate,
        implied_volatility,
        final_steps,
    ).reshape(pre_second.shape)
    option = np.maximum(terminal, sign * (pre_second - strike))

    for level in range(middle_steps - 1, -1, -1):
        continuation = discount * (
            probability * option[:, 1:] + (1.0 - probability) * option[:, :-1]
        )
        nodes = np.arange(level + 1)
        prices = ex_first[:, None] * up**nodes * down ** (level - nodes)
        option = np.maximum(continuation, sign * (prices - strike))
    first_dividend_continuation = option[:, 0]
    option = np.maximum(first_dividend_continuation, sign * (pre_first - strike))

    for level in range(first_steps - 1, -1, -1):
        continuation = discount * (
            probability * option[1:] + (1.0 - probability) * option[:-1]
        )
        nodes = np.arange(level + 1)
        prices = underlying * up**nodes * down ** (level - nodes)
        option = np.maximum(continuation, sign * (prices - strike))
    return float(option[0])


def _american_discrete_recursive(
    call: bool,
    underlying: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_amounts: NDArray[np.float64],
    dividend_steps: NDArray[np.int64],
    implied_volatility: float,
    steps: int,
) -> float:
    """Literal reference recursion for zero, one, or uncommon dividend counts."""

    if dividend_amounts.size == 0 or dividend_steps[0] > steps:
        return american_continuous(
            call, underlying, strike, time_to_maturity, risk_free_rate,
            risk_free_rate, implied_volatility, steps
        )
    dt = time_to_maturity / steps
    up = np.exp(implied_volatility * np.sqrt(dt))
    down = 1.0 / up
    probability = (np.exp(risk_free_rate * dt) - down) / (up - down)
    discount = np.exp(-risk_free_rate * dt)
    sign = 1.0 if call else -1.0
    first = int(dividend_steps[0])
    option = np.empty(first + 1)
    for node in range(first + 1):
        price = underlying * up**node * down ** (first - node)
        continuation = _american_discrete_recursive(
            call, price - dividend_amounts[0], strike,
            time_to_maturity - first * dt, risk_free_rate,
            dividend_amounts[1:], dividend_steps[1:] - first,
            implied_volatility, steps - first,
        )
        option[node] = max(continuation, sign * (price - strike), 0.0)
    for level in range(first - 1, -1, -1):
        continuation = discount * (
            probability * option[1:] + (1.0 - probability) * option[:-1]
        )
        nodes = np.arange(level + 1)
        prices = underlying * up**nodes * down ** (level - nodes)
        option = np.maximum(continuation, sign * (prices - strike))
    return float(option[0])
