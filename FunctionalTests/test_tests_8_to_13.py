"""Professor-data functional tests for Tests 8.1 through 13.9.

How this file works
-------------------
1. Read only professor-provided input CSV files from ``FunctionalTests/data``.
2. Pass those inputs to reusable functions in ``risk_management`` and compute
   an independent ``actual`` result.
3. Read each professor ``testout*.csv`` only after the calculation, as the
   ``expected`` test oracle.
4. Display actual, expected, and actual-minus-expected through the same helpers
   used by Tests 1.1-7.6 when detailed output mode is enabled.

No function in ``risk_management`` reads a ``testout`` file. Deterministic
tests use tight numerical comparisons. Simulation tests use tail-estimator
standard errors because Julia and NumPy produce different random streams even
with the same seed; a separate relative guardrail is retained as well.

Test coverage
-------------
- 8.1-8.6: parametric and simulated VaR / Expected Shortfall.
- 9.1: Gaussian-copula portfolio VaR / Expected Shortfall.
- 10.1-10.4: risk parity and maximum-Sharpe portfolio optimization.
- 11.1-11.2: ex-post return and volatility attribution.
- 12.1-12.3: European and American option valuation and Greeks.
- 13.1-13.4: Kendall correlation and multivariate-t fitting.
- 13.5-13.8: Gaussian/t copulas, model selection, and tail dependence.
- 13.9: t-copula portfolio VaR / Expected Shortfall.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from risk_management import (
    aggregate_portfolio_risk,
    american_discrete_dividends,
    american_finite_difference_greeks,
    copula_aicc,
    copula_bic,
    expected_shortfall,
    expost_factor,
    fit_gaussian_copula,
    fit_multivariate_t,
    fit_normal,
    fit_student_t,
    fit_t_copula,
    gbsm,
    kendall_correlation,
    maximize_sharpe_ratio,
    risk_parity,
    scipy_distribution,
    simulate_fitted_distribution,
    simulate_gaussian_copula,
    simulate_t_copula,
    tail_dependence_t,
    value_at_risk,
)

from test_tests_1_to_7 import (
    DATA,
    SHOW_TEST_OUTPUTS,
    assert_dataframes_close,
    assert_numeric_close,
    matrix,
    report_comparison,
)


TAIL_ALPHA = 0.05
TAIL_SIMULATION_SIGMA_LIMIT = 4.0
TAIL_SIMULATION_RELATIVE_LIMIT_AT_100K = 0.05


def _distribution_risk_frame(model, prefix: str) -> pd.DataFrame:
    """Build the two-column Test 8 result from a fitted distribution."""

    centered = (
        stats.norm(loc=0.0, scale=model.sigma)
        if hasattr(model, "nu") is False
        else stats.t(df=model.nu, loc=0.0, scale=model.sigma)
    )
    risk_function = value_at_risk if prefix == "VaR" else expected_shortfall
    return pd.DataFrame(
        {
            f"{prefix} Absolute": [risk_function(model)],
            f"{prefix} Diff from Mean": [risk_function(centered)],
        }
    )


def _simulation_risk_frame(values: np.ndarray, prefix: str) -> pd.DataFrame:
    risk_function = value_at_risk if prefix == "VaR" else expected_shortfall
    return pd.DataFrame(
        {
            f"{prefix} Absolute": [risk_function(values)],
            f"{prefix} Diff from Mean": [risk_function(values - values.mean())],
        }
    )


def _tail_standard_errors(values: np.ndarray, alpha: float = TAIL_ALPHA):
    """Asymptotic standard errors for empirical VaR and Expected Shortfall."""

    pnl = np.asarray(values, dtype=float).reshape(-1)
    cutoff = -value_at_risk(pnl, alpha)
    density = float(stats.gaussian_kde(pnl)([cutoff])[0])
    var_standard_error = np.sqrt(alpha * (1.0 - alpha) / pnl.size) / density

    lower_mean = float(np.mean(pnl[pnl <= cutoff]))
    influence = (
        cutoff
        - lower_mean
        - np.where(pnl <= cutoff, (cutoff - pnl) / alpha, 0.0)
    )
    es_standard_error = float(np.std(influence, ddof=1) / np.sqrt(pnl.size))
    return var_standard_error, es_standard_error


def assert_simulated_risk_close(
    test: unittest.TestCase,
    test_id: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    pnl_by_row: list[np.ndarray],
) -> None:
    """Validate independent Monte Carlo tail estimates statistically.

    The actual and professor results are independent simulations. Their
    difference therefore has approximately twice one run's variance. Each
    VaR/ES difference must be within four combined standard errors. The second
    guardrail is five percent at 100,000 draws and scales by ``1/sqrt(n)`` for
    smaller simulations, following the sampling-error rate rather than using
    an arbitrary fixed relaxation. ES must also remain at least VaR.
    """

    report_comparison(test_id, actual, expected)
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(list(actual.columns), list(expected.columns))
    if "Stock" in actual:
        test.assertEqual(actual["Stock"].tolist(), expected["Stock"].tolist())

    if "VaR95" in actual:
        comparisons = [
            (row, column, pnl_by_row[row], error_index)
            for row in range(len(pnl_by_row))
            for column, error_index in [("VaR95", 0), ("ES95", 1)]
        ]
    else:
        error_index = 0 if actual.columns[0].startswith("VaR") else 1
        comparisons = [
            (0, column, pnl_by_row[index], error_index)
            for index, column in enumerate(actual.columns)
        ]

    for row, column, pnl, error_index in comparisons:
        standard_errors = _tail_standard_errors(pnl)
        difference = abs(float(actual.loc[row, column] - expected.loc[row, column]))
        combined_error = np.sqrt(2.0) * standard_errors[error_index]
        z_score = difference / combined_error
        relative_difference = difference / abs(float(expected.loc[row, column]))
        relative_limit = TAIL_SIMULATION_RELATIVE_LIMIT_AT_100K * np.sqrt(
            100_000.0 / len(pnl)
        )
        if SHOW_TEST_OUTPUTS:
            print(
                f"  {column} row {row}: difference={difference:.12g}, "
                f"combined SE={combined_error:.12g}, z={z_score:.6g}, "
                f"relative={100.0 * relative_difference:.6g}% "
                f"(limit={100.0 * relative_limit:.6g}%)"
            )
        test.assertLessEqual(
            z_score,
            TAIL_SIMULATION_SIGMA_LIMIT,
            msg=f"Test {test_id} {column} row {row} exceeds four Monte Carlo SEs",
        )
        test.assertLessEqual(
            relative_difference,
            relative_limit,
            msg=f"Test {test_id} {column} row {row} exceeds the 1/sqrt(n) guardrail",
        )

    if "ES95" in actual:
        test.assertTrue(np.all(actual["ES95"] >= actual["VaR95"]))


class TestTailRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.normal_values = matrix("test7_1.csv")[:, 0]
        cls.t_values = matrix("test7_2.csv")[:, 0]
        cls.normal_fit = fit_normal(cls.normal_values)
        cls.t_fit = fit_student_t(cls.t_values)

    def test_8_1_var_normal(self) -> None:
        actual = _distribution_risk_frame(self.normal_fit, "VaR")
        expected = pd.read_csv(DATA / "testout8_1.csv")
        assert_dataframes_close(
            "8.1", actual, expected, check_exact=False, rtol=1e-10, atol=1e-12
        )

    def test_8_2_var_student_t(self) -> None:
        actual = _distribution_risk_frame(self.t_fit, "VaR")
        expected = pd.read_csv(DATA / "testout8_2.csv")
        assert_dataframes_close(
            "8.2", actual, expected, check_exact=False, rtol=2e-5, atol=1e-8
        )

    def test_8_3_var_simulation(self) -> None:
        simulation = simulate_fitted_distribution(self.t_fit, 10_000, seed=8)
        actual = _simulation_risk_frame(simulation, "VaR")
        expected = pd.read_csv(DATA / "testout8_3.csv")
        assert_simulated_risk_close(
            self,
            "8.3",
            actual,
            expected,
            [simulation, simulation - simulation.mean()],
        )
        # Simulation must independently agree with its fitted t model.
        analytic = _distribution_risk_frame(self.t_fit, "VaR")
        self.assertLess(
            abs(actual.iloc[0, 0] - analytic.iloc[0, 0]),
            4.0 * _tail_standard_errors(simulation)[0],
        )

    def test_8_4_es_normal(self) -> None:
        actual = _distribution_risk_frame(self.normal_fit, "ES")
        expected = pd.read_csv(DATA / "testout8_4.csv")
        assert_dataframes_close(
            "8.4", actual, expected, check_exact=False, rtol=1e-10, atol=1e-12
        )

    def test_8_5_es_student_t(self) -> None:
        actual = _distribution_risk_frame(self.t_fit, "ES")
        expected = pd.read_csv(DATA / "testout8_5.csv")
        assert_dataframes_close(
            "8.5", actual, expected, check_exact=False, rtol=2e-5, atol=1e-8
        )

    def test_8_6_es_simulation(self) -> None:
        rng = np.random.default_rng(8)
        simulate_fitted_distribution(self.t_fit, 10_000, rng=rng)
        simulation = simulate_fitted_distribution(self.t_fit, 10_000, rng=rng)
        actual = _simulation_risk_frame(simulation, "ES")
        expected = pd.read_csv(DATA / "testout8_6.csv")
        assert_simulated_risk_close(
            self,
            "8.6",
            actual,
            expected,
            [simulation, simulation - simulation.mean()],
        )
        analytic = _distribution_risk_frame(self.t_fit, "ES")
        self.assertLess(
            abs(actual.iloc[0, 0] - analytic.iloc[0, 0]),
            4.0 * _tail_standard_errors(simulation)[1],
        )


class TestGaussianCopulaPortfolioRisk(unittest.TestCase):
    def test_9_1_gaussian_copula_var_es(self) -> None:
        returns = pd.read_csv(DATA / "test9_1_returns.csv")
        portfolio = pd.read_csv(DATA / "test9_1_portfolio.csv", encoding="utf-8-sig")
        fits = [fit_normal(returns["A"]), fit_student_t(returns["B"])]
        uniforms = np.column_stack(
            [scipy_distribution(fit).cdf(returns[column]) for fit, column in zip(fits, returns)]
        )
        spearman = float(stats.spearmanr(uniforms[:, 0], uniforms[:, 1]).statistic)
        correlation = np.array([[1.0, spearman], [spearman, 1.0]])
        simulated_uniforms = simulate_gaussian_copula(
            correlation, 100_000, seed=1234
        )
        simulated_returns = np.column_stack(
            [
                scipy_distribution(fit).ppf(simulated_uniforms[:, column])
                for column, fit in enumerate(fits)
            ]
        )
        current_values = (
            portfolio["Holding"].to_numpy(float)
            * portfolio["Starting Price"].to_numpy(float)
        )
        names = portfolio["Stock"].astype(str).tolist()
        actual = aggregate_portfolio_risk(simulated_returns, current_values, names)
        expected = pd.read_csv(DATA / "testout9_1.csv")
        pnl = simulated_returns * current_values
        assert_simulated_risk_close(
            self,
            "9.1",
            actual,
            expected,
            [pnl[:, 0], pnl[:, 1], pnl.sum(axis=1)],
        )
        np.testing.assert_allclose(
            actual["VaR95_Pct"], actual["VaR95"] / np.r_[current_values, current_values.sum()]
        )
        np.testing.assert_allclose(
            actual["ES95_Pct"], actual["ES95"] / np.r_[current_values, current_values.sum()]
        )


class TestPortfolioOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.risk_covariance = matrix("test5_2.csv")
        cls.sharpe_covariance = matrix("test5_3.csv")
        cls.means = matrix("test10_3_means.csv")[:, 0]

    def test_10_1_equal_risk_parity(self) -> None:
        actual = risk_parity(self.risk_covariance)
        expected = matrix("testout10_1.csv")[:, 0]
        assert_numeric_close("10.1", actual, expected, rtol=2e-6, atol=1e-8)

    def test_10_2_budgeted_risk_parity(self) -> None:
        actual = risk_parity(self.risk_covariance, [1, 1, 1, 1, 0.5])
        expected = matrix("testout10_2.csv")[:, 0]
        assert_numeric_close("10.2", actual, expected, rtol=2e-6, atol=1e-8)

    def test_10_3_maximum_sharpe_long_only(self) -> None:
        actual = maximize_sharpe_ratio(
            self.sharpe_covariance, self.means, 0.04
        )
        expected = matrix("testout10_3.csv")[:, 0]
        assert_numeric_close("10.3", actual, expected, rtol=1e-5, atol=2e-6)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=10)
        self.assertGreaterEqual(float(actual.min()), -1e-10)

    def test_10_4_maximum_sharpe_bounded(self) -> None:
        bounds = np.tile([0.1, 0.5], (5, 1))
        actual = maximize_sharpe_ratio(
            self.sharpe_covariance, self.means, 0.04, bounds
        )
        expected = matrix("testout10_4.csv")[:, 0]
        assert_numeric_close("10.4", actual, expected, rtol=1e-6, atol=3e-8)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=10)


class TestExPostAttribution(unittest.TestCase):
    def test_11_1_asset_attribution(self) -> None:
        returns = pd.read_csv(DATA / "test11_1_returns.csv")
        weights = matrix("test11_1_weights.csv")[:, 0]
        actual = expost_factor(
            weights, returns, returns, np.eye(returns.shape[1])
        ).attribution.drop(columns="Alpha")
        expected = pd.read_csv(DATA / "testout11_1.csv")
        assert_dataframes_close(
            "11.1", actual, expected, check_exact=False, rtol=1e-10, atol=1e-12
        )

    def test_11_2_factor_attribution(self) -> None:
        factor_returns = pd.read_csv(DATA / "test11_2_factor_returns.csv")
        stock_returns = pd.read_csv(DATA / "test11_2_stock_returns.csv")
        betas = pd.read_csv(DATA / "test11_2_beta.csv").iloc[:, 1:].to_numpy(float)
        weights = matrix("test11_2_weights.csv")[:, 0]
        actual = expost_factor(
            weights, stock_returns, factor_returns, betas
        ).attribution
        expected = pd.read_csv(DATA / "testout11_2.csv")
        assert_dataframes_close(
            "11.2", actual, expected, check_exact=False, rtol=1e-10, atol=1e-12
        )


class TestOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.options = pd.read_csv(DATA / "test12_1.csv").dropna(subset=["ID"])

    def test_12_1_european_gbsm_greeks(self) -> None:
        rows = []
        for _, option in self.options.iterrows():
            result = gbsm(
                option["Option Type"] == "Call",
                option.Underlying,
                option.Strike,
                option.DaysToMaturity / option.DayPerYear,
                option.RiskFreeRate,
                option.RiskFreeRate - option.DividendRate,
                option.ImpliedVol,
                include_greeks=True,
            )
            rows.append(
                [int(option.ID), result.value, result.delta, result.gamma, result.vega, result.rho, result.theta]
            )
        actual = pd.DataFrame(rows, columns=["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"])
        expected = pd.read_csv(DATA / "testout12_1.csv")
        assert_dataframes_close(
            "12.1", actual, expected, check_exact=False, rtol=1e-10, atol=1e-12
        )

    def test_12_2_american_continuous_greeks(self) -> None:
        rows = []
        for _, option in self.options.iterrows():
            result = american_finite_difference_greeks(
                option["Option Type"] == "Call",
                option.Underlying,
                option.Strike,
                option.DaysToMaturity / option.DayPerYear,
                option.RiskFreeRate,
                option.RiskFreeRate - option.DividendRate,
                option.ImpliedVol,
                steps=500,
            )
            rows.append(
                [int(option.ID), result.value, result.delta, result.gamma, result.vega, result.rho, result.theta]
            )
        actual = pd.DataFrame(rows, columns=["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"])
        expected = pd.read_csv(DATA / "testout12_2.csv")
        assert_dataframes_close(
            "12.2", actual, expected, check_exact=False, rtol=2e-8, atol=2e-9
        )

    def test_12_3_american_discrete_dividends(self) -> None:
        rows = []
        options = pd.read_csv(DATA / "test12_3.csv")
        for _, option in options.iterrows():
            dividend_amounts = [float(value) for value in option.DividendAmts.split(",")]
            dividend_steps = [2 * int(value) for value in option.DividendDates.split(",")]
            value = american_discrete_dividends(
                option["Option Type"] == "Call",
                option.Underlying,
                option.Strike,
                option.DaysToMaturity / option.DayPerYear,
                option.RiskFreeRate,
                dividend_amounts,
                dividend_steps,
                option.ImpliedVol,
                int(option.DaysToMaturity * 2),
            )
            rows.append([option.ID, value])
        actual = pd.DataFrame(rows, columns=["ID", "Value"])
        expected = pd.read_csv(DATA / "testout12_3.csv")
        assert_dataframes_close(
            "12.3", actual, expected, check_exact=False, rtol=1e-10, atol=1e-12
        )


class TestMultivariateTAndCopulas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.returns_frame = pd.read_csv(DATA / "test13_returns.csv")
        cls.returns = cls.returns_frame.to_numpy(float)
        cls.multivariate_fit = fit_multivariate_t(cls.returns)
        cls.margin_fits = [
            fit_student_t(cls.returns[:, column])
            for column in range(cls.returns.shape[1])
        ]
        cls.uniforms = np.column_stack(
            [
                scipy_distribution(fit).cdf(cls.returns[:, column])
                for column, fit in enumerate(cls.margin_fits)
            ]
        )
        cls.gaussian_correlation, cls.gaussian_ll = fit_gaussian_copula(cls.uniforms)
        cls.t_correlation, cls.copula_nu, cls.t_ll = fit_t_copula(cls.uniforms)

    def test_13_1_kendall_correlation(self) -> None:
        actual = kendall_correlation(self.returns)
        expected = matrix("testout13_1.csv")
        assert_numeric_close("13.1", actual, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.diag(actual), 1.0, rtol=0.0, atol=1e-12)
        self.assertGreater(float(np.linalg.eigvalsh(actual).min()), 0.0)

    def test_13_2_multivariate_t_mean(self) -> None:
        expected = matrix("testout13_2.csv")[:, 0]
        assert_numeric_close(
            "13.2", self.multivariate_fit.mean, expected, rtol=1e-10, atol=1e-12
        )

    def test_13_3_multivariate_t_scale(self) -> None:
        expected = matrix("testout13_3.csv")
        assert_numeric_close(
            "13.3", self.multivariate_fit.scale, expected, rtol=1e-10, atol=1e-12
        )

    def test_13_4_multivariate_t_nu_likelihood(self) -> None:
        actual = [self.multivariate_fit.nu, self.multivariate_fit.log_likelihood]
        expected = matrix("testout13_4.csv")[0]
        assert_numeric_close("13.4", actual, expected, rtol=1e-10, atol=1e-10)

    def test_13_5_gaussian_copula_likelihood(self) -> None:
        expected = float(matrix("testout13_5.csv")[0, 0])
        assert_numeric_close(
            "13.5", self.gaussian_ll, expected, rtol=3e-7, atol=2e-5
        )

    def test_13_6_t_copula_nu_likelihood(self) -> None:
        actual = [self.copula_nu, self.t_ll]
        expected = matrix("testout13_6.csv")[0]
        assert_numeric_close("13.6", actual, expected, rtol=5e-7, atol=5e-5)

    def test_13_7_copula_model_selection(self) -> None:
        observations = self.returns.shape[0]
        actual = pd.DataFrame(
            {
                "Copula": ["Gaussian", "T"],
                "LL": [self.gaussian_ll, self.t_ll],
                "K": [0, 1],
                "AICC": [
                    copula_aicc(self.gaussian_ll, 0, observations),
                    copula_aicc(self.t_ll, 1, observations),
                ],
                "BIC": [
                    copula_bic(self.gaussian_ll, 0, observations),
                    copula_bic(self.t_ll, 1, observations),
                ],
            }
        )
        expected = pd.read_csv(DATA / "testout13_7.csv")
        assert_dataframes_close(
            "13.7", actual, expected, check_exact=False, rtol=5e-7, atol=1e-4
        )
        self.assertLess(float(actual.loc[1, "AICC"]), float(actual.loc[0, "AICC"]))
        self.assertLess(float(actual.loc[1, "BIC"]), float(actual.loc[0, "BIC"]))

    def test_13_8_t_copula_tail_dependence(self) -> None:
        rows = []
        n_assets = self.t_correlation.shape[0]
        for row in range(n_assets):
            for column in range(row + 1, n_assets):
                rho = float(self.t_correlation[row, column])
                rows.append(
                    [row + 1, column + 1, rho, tail_dependence_t(rho, self.copula_nu)]
                )
        actual = pd.DataFrame(rows, columns=["I", "J", "Rho", "Lambda"])
        expected = pd.read_csv(DATA / "testout13_8.csv")
        assert_dataframes_close(
            "13.8", actual, expected, check_exact=False, rtol=5e-7, atol=5e-8
        )
        self.assertTrue(np.all((actual["Lambda"] > 0.0) & (actual["Lambda"] < 1.0)))

    def test_13_9_t_copula_portfolio_var_es(self) -> None:
        portfolio = pd.read_csv(DATA / "test13_portfolio.csv")
        simulated_uniforms = simulate_t_copula(
            self.t_correlation, self.copula_nu, 100_000, seed=13
        )
        simulated_returns = np.column_stack(
            [
                scipy_distribution(fit).ppf(simulated_uniforms[:, column])
                for column, fit in enumerate(self.margin_fits)
            ]
        )
        current_values = portfolio["currentValue"].to_numpy(float)
        names = portfolio["Stock"].astype(str).tolist()
        actual = aggregate_portfolio_risk(simulated_returns, current_values, names)
        expected = pd.read_csv(DATA / "testout13_9.csv")
        pnl = simulated_returns * current_values
        pnl_rows = [pnl[:, column] for column in range(pnl.shape[1])]
        pnl_rows.append(pnl.sum(axis=1))
        assert_simulated_risk_close(
            self, "13.9", actual, expected, pnl_rows
        )
        np.testing.assert_allclose(
            actual["VaR95_Pct"], actual["VaR95"] / np.r_[current_values, current_values.sum()]
        )
        np.testing.assert_allclose(
            actual["ES95_Pct"], actual["ES95"] / np.r_[current_values, current_values.sum()]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
