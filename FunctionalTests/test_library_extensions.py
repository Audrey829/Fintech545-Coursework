import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_management import (  # noqa: E402
    NormalFit,
    StudentTFit,
    calculate_returns,
    complete_observation_count,
    copula_log_likelihood_contributions,
    copula_risk_differences,
    correlation_diagnostics,
    covariance_from_correlation,
    delta_normal_var,
    ew_covariance,
    ewma_diagnostics,
    fit_best_margins,
    fit_copula_models,
    fit_market_model,
    gaussian_copula_log_likelihood,
    joint_observation_counts,
    kurtosis_outlier_sensitivity,
    largest_matrix_movements,
    normal_scale_mixture_excess_kurtosis,
    pnl_from_return_scenarios,
    portfolio_variance,
    rank_uniforms,
    repair_correlation_summary,
    sample_standard_deviations,
    sample_moments,
    simulate_copula_returns,
    simulate_market_model,
    smallest_eigenvector_loadings,
    t_copula_log_likelihood,
    tail_count_model_comparison,
    threshold_event_counts,
    uniforms_from_fitted_margins,
    var_sampling_noise,
)


class TestReusableAnalyticsExtensions(unittest.TestCase):
    def test_joint_observation_counts(self):
        frame = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [1.0, np.nan, 3.0]})
        counts = joint_observation_counts(frame)
        self.assertEqual(int(counts.loc["A", "A"]), 2)
        self.assertEqual(int(counts.loc["A", "B"]), 1)
        self.assertEqual(complete_observation_count(frame), 1)

    def test_correlation_diagnostics(self):
        frame = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [2.0, 1.0, 4.0, 3.0],
                "C": [1.0, 2.0, np.nan, 5.0],
            }
        )
        diagnostics = correlation_diagnostics(frame)
        self.assertEqual(diagnostics["complete"].shape, (3, 3))
        self.assertEqual(diagnostics["pairwise"].shape, (3, 3))

    def test_covariance_and_portfolio_variance(self):
        correlation = np.array([[1.0, 0.25], [0.25, 1.0]])
        covariance = covariance_from_correlation(correlation, [2.0, 3.0])
        np.testing.assert_allclose(covariance, [[4.0, 1.5], [1.5, 9.0]])
        self.assertAlmostEqual(portfolio_variance(covariance, [1.0, -1.0]), 10.0)

    def test_sample_standard_deviations(self):
        values = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 8.0]])
        np.testing.assert_allclose(
            sample_standard_deviations(values, skip_missing=True),
            [2.0, np.sqrt(18.0)],
        )
        with self.assertRaises(ValueError):
            sample_standard_deviations(values)

    def test_psd_repair_diagnostics(self):
        original = np.array([[1.0, 1.2], [1.2, 1.0]])
        summary = repair_correlation_summary(original)
        self.assertEqual(set(summary), {"Rebonato-Jackel", "Higham"})
        for result in summary.values():
            self.assertGreaterEqual(result["minimum_eigenvalue"], -1e-8)
            self.assertGreater(result["frobenius_distance"], 0.0)
        movements = largest_matrix_movements(
            np.eye(3),
            np.array([[1.0, 0.2, -0.4], [0.2, 1.0, 0.1], [-0.4, 0.1, 1.0]]),
            ["A", "B", "C"],
            top_n=1,
        )
        self.assertEqual(movements.loc[0, "Pair"], "A-C")
        self.assertAlmostEqual(movements.loc[0, "Absolute Change"], 0.4)

    def test_smallest_eigenvector_loadings(self):
        loadings = smallest_eigenvector_loadings(
            np.diag([0.5, 2.0, 3.0]), ["A", "B", "C"]
        )
        self.assertEqual(loadings.loc[0, "Series"], "A")
        self.assertAlmostEqual(loadings.loc[0, "Absolute Loading"], 1.0)
        self.assertAlmostEqual(loadings.loc[0, "Smallest Eigenvalue"], 0.5)

    def test_calculate_returns_can_demean(self):
        prices = pd.DataFrame({"Day": [1, 2, 3], "A": [100.0, 110.0, 99.0]})
        returns = calculate_returns(
            prices, method="DISCRETE", date_column="Day", demean=True
        )
        self.assertAlmostEqual(float(returns["A"].mean()), 0.0)

    def test_descriptive_statistics(self):
        moments = sample_moments([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(moments["Mean"], 3.0)
        regimes = pd.DataFrame(
            {
                "Observations": [50, 50],
                "Standard Deviation": [1.0, 2.0],
            }
        )
        self.assertAlmostEqual(normal_scale_mixture_excess_kurtosis(regimes), 1.08)

    def test_kurtosis_outlier_sensitivity(self):
        frame = pd.DataFrame({"X": [-2.0, -1.0, 0.0, 1.0, 2.0, 15.0]})
        result = kurtosis_outlier_sensitivity(frame)
        self.assertEqual(int(result.loc[0, "Most Influential Row Index"]), 5)
        self.assertLess(
            result.loc[0, "Excess Kurtosis Without It"],
            result.loc[0, "Full Excess Kurtosis"],
        )

    def test_threshold_event_counts(self):
        values = pd.DataFrame({"A": [-2.0, 1.0, -3.0], "B": [0.0, -2.0, -4.0]})
        counts = threshold_event_counts(values, threshold=-1.0)
        self.assertEqual(counts, {"A": 2, "B": 2, "At least one": 3, "All": 1})

    def test_rank_uniforms_are_inside_unit_interval(self):
        uniforms = rank_uniforms(pd.DataFrame({"A": [3.0, 1.0, 2.0]}))
        self.assertTrue(((uniforms > 0.0) & (uniforms < 1.0)).all().all())

    def test_tail_count_model_comparison(self):
        empirical = pd.DataFrame(
            {"Pair": ["A-B"], "Worst Tail Count": [4], "Best Tail Count": [6]}
        )
        gaussian = pd.DataFrame(
            {
                "Pair": ["A-B"],
                "Worst Tail Count (scaled)": [3.5],
                "Best Tail Count (scaled)": [7.0],
            }
        )
        student = pd.DataFrame(
            {
                "Pair": ["A-B"],
                "Worst Tail Count (scaled)": [4.2],
                "Best Tail Count (scaled)": [5.8],
            }
        )
        comparison = tail_count_model_comparison(empirical, gaussian, student)
        np.testing.assert_allclose(
            comparison["Gaussian Absolute Error"], [0.5, 1.0]
        )
        np.testing.assert_allclose(
            comparison["Student-t Absolute Error"], [0.2, 0.2]
        )

    def test_copula_log_likelihood_contributions_reconcile(self):
        uniforms = pd.DataFrame(
            {"A": [0.02, 0.25, 0.60, 0.98], "B": [0.03, 0.40, 0.70, 0.97]}
        )
        gaussian_correlation = np.array([[1.0, 0.35], [0.35, 1.0]])
        t_correlation = np.array([[1.0, 0.45], [0.45, 1.0]])
        fitted = {
            "gaussian_correlation": gaussian_correlation,
            "t_correlation": t_correlation,
            "t_nu": 5.0,
        }
        contributions = copula_log_likelihood_contributions(uniforms, fitted)
        expected = t_copula_log_likelihood(uniforms, t_correlation, 5.0) - (
            gaussian_copula_log_likelihood(uniforms, gaussian_correlation)
        )
        self.assertAlmostEqual(contributions["t minus Gaussian"].sum(), expected)
        self.assertEqual(
            contributions["No series in outer 5%"].tolist(),
            [False, True, True, False],
        )

    def test_margin_and_copula_pipeline(self):
        rng = np.random.default_rng(545)
        frame = pd.DataFrame(
            {
                "Normal": rng.normal(size=250),
                "Heavy": rng.standard_t(df=3.5, size=250),
            }
        )
        margin_table, margins = fit_best_margins(frame)
        self.assertEqual(set(margin_table["Series"]), set(frame.columns))
        self.assertIsInstance(margins["Normal"], (NormalFit, StudentTFit))
        self.assertIsInstance(margins["Heavy"], StudentTFit)
        uniforms = uniforms_from_fitted_margins(frame, margins)
        self.assertTrue(((uniforms > 0.0) & (uniforms < 1.0)).all().all())
        copula_table, fitted_copulas = fit_copula_models(uniforms)
        self.assertEqual(copula_table["Copula"].tolist(), ["Gaussian", "Student-t"])
        simulated = simulate_copula_returns(
            fitted_copulas, margins, list(frame.columns), n_samples=100, seed=545
        )
        self.assertEqual(simulated["gaussian_returns"].shape, (100, 2))
        self.assertEqual(simulated["t_returns"].shape, (100, 2))
        self.assertTrue(np.isfinite(simulated["t_returns"]).all())

    def test_copula_risk_differences(self):
        table = pd.DataFrame(
            [
                {"Source": "Gaussian copula", "Alpha": 0.05, "VaR ($)": 10.0, "ES ($)": 20.0},
                {"Source": "Student-t copula", "Alpha": 0.05, "VaR ($)": 13.0, "ES ($)": 26.0},
            ]
        )
        difference = copula_risk_differences(table)
        self.assertAlmostEqual(difference.loc[0, "t minus Gaussian VaR ($)"], 3.0)
        self.assertAlmostEqual(difference.loc[0, "t minus Gaussian ES ($)"], 6.0)

    def test_market_model_and_delta_normal_var(self):
        market = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        asset = 0.001 + 1.5 * market + np.array([0.001, -0.001, 0.0, 0.001, -0.001])
        fit = fit_market_model(asset, market)
        self.assertAlmostEqual(fit.beta, 1.48, places=10)
        result = delta_normal_var(
            np.array([[0.04, 0.01], [0.01, 0.09]]), {"P": [1.0, 1.0]}
        )
        self.assertGreater(float(result.loc[0, "Delta-normal VaR ($)"]), 0.0)

    def test_market_model_simulation_covariance_modes(self):
        market = np.array([-0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
        asset_a = 0.001 + 1.2 * market + np.array([0.002, -0.001, 0.0, 0.001, -0.002, 0.0])
        asset_b = -0.001 - 0.5 * market + np.array([0.001, -0.002, 0.001, 0.002, -0.001, -0.001])
        fits = {
            "A": fit_market_model(asset_a, market),
            "B": fit_market_model(asset_b, market),
        }
        correlated, full_covariance = simulate_market_model(
            market,
            fits,
            n_samples=50,
            seed=545,
            include_residual_correlation=True,
        )
        independent, diagonal_covariance = simulate_market_model(
            market,
            fits,
            n_samples=50,
            seed=545,
            include_residual_correlation=False,
        )
        self.assertEqual(correlated.shape, (50, 2))
        self.assertEqual(independent.shape, (50, 2))
        self.assertNotEqual(float(full_covariance[0, 1]), 0.0)
        self.assertAlmostEqual(float(diagonal_covariance[0, 1]), 0.0)

    def test_ewma_and_var_sampling_noise(self):
        values = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        diagnostics = ewma_diagnostics(values, 0.94, recent_days=2)
        expected_volatility = np.sqrt(ew_covariance(values[:, None], 0.94)[0, 0])
        self.assertAlmostEqual(diagnostics["Volatility"], expected_volatility)
        noise = var_sampling_noise(
            0.02, 50.0, position_value=1_000_000.0, alpha=0.05
        )
        self.assertAlmostEqual(noise["Volatility Standard Error"], 0.002)
        self.assertAlmostEqual(noise["Dollar Volatility Standard Error"], 2000.0)
        self.assertGreater(noise["Dollar VaR Standard Error"], 3000.0)

    def test_pnl_from_return_scenarios(self):
        returns = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.01]})
        pnl = pnl_from_return_scenarios(
            returns, {"Long-short": {"A": 100.0, "B": -50.0}}
        )
        np.testing.assert_allclose(pnl["Long-short"], [-0.5, -2.5])

    def test_repository_and_submission_library_are_synchronized(self):
        repository_root = Path(__file__).resolve().parents[1]
        canonical = repository_root / "risk_management"
        submission = repository_root / "Assignment2" / "risk_management"
        source_names = sorted(path.name for path in canonical.glob("*.py"))
        self.assertEqual(source_names, sorted(path.name for path in submission.glob("*.py")))
        for name in source_names:
            self.assertEqual(
                (canonical / name).read_bytes(),
                (submission / name).read_bytes(),
                msg=f"library copy differs: {name}",
            )
        self.assertEqual(
            (canonical / "README.md").read_bytes(),
            (submission / "README.md").read_bytes(),
            msg="library copy differs: README.md",
        )


if __name__ == "__main__":
    unittest.main()
