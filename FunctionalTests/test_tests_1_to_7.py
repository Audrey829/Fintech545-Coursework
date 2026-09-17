"""Professor-data functional tests for Tests 1.1 through 7.6.

How this file works
-------------------
1. Read the professor-provided input CSV files from ``FunctionalTests/data``.
2. Pass those inputs to functions implemented in the ``risk_management``
   package and calculate an ``actual`` result.
3. Read the matching professor-provided ``testout*.csv`` file only as the
   ``expected`` result, then compare ``actual`` with ``expected``.

The calculation functions do not read the expected-output files. They operate
only on the arrays or DataFrames passed to them. The ``testout`` files are test
oracles used by this test module to determine whether a calculation passes.

To display every calculated result beside the professor's expected result,
run ``python FunctionalTests/run_tests_with_outputs.py`` from the repository
root. That detailed mode also prints ``actual - expected``, the maximum
absolute difference, and the mean absolute difference.

Test coverage
-------------
- 1.1-1.4: covariance and correlation with missing values.
- 2.1-2.3: exponentially weighted covariance and correlation.
- 3.1-3.4: near-PSD and Higham PSD matrix repair.
- 4.1: positive-semidefinite Cholesky factorization.
- 5.1-5.5: multivariate Normal and PCA simulation.
- 6.1-6.2: arithmetic and logarithmic returns.
- 7.1-7.4: Normal fit, Student-t fit, t-error regression, and AICc.
- 7.5: Normal Inverse Gaussian fit by the method of moments.
- 7.6: Normal Inverse Gaussian fit by maximum likelihood.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import stats

from risk_management import (
    aicc,
    calculate_returns,
    chol_psd,
    covariance_with_ew_variance,
    ew_covariance,
    fit_normal,
    fit_nig_mle,
    fit_nig_moments,
    fit_student_t,
    fit_t_regression,
    higham_nearest_psd,
    missing_covariance,
    near_psd,
    simulate_normal,
    simulate_pca,
)


DATA = Path(__file__).resolve().parent / "data"
SHOW_TEST_OUTPUTS = os.environ.get("SHOW_TEST_OUTPUTS", "0") == "1"
SIMULATION_SIGMA_LIMIT = 4.0
SIMULATION_SCALE_LIMIT = 0.025


def matrix(filename: str) -> np.ndarray:
    return pd.read_csv(DATA / filename).to_numpy(dtype=float)


def _preview_array(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.ndim == 0:
        return f"{float(array):.12g}"
    if array.size <= 36:
        preview = array
    elif array.ndim == 1:
        preview = array[:12]
    else:
        preview = array[:3, : min(6, array.shape[1])]
    return np.array2string(
        preview,
        precision=10,
        suppress_small=False,
        max_line_width=140,
    )


def report_comparison(test_id: str, actual, expected) -> None:
    """Print actual, expected, and difference when detailed mode is enabled."""

    if not SHOW_TEST_OUTPUTS:
        return

    print(f"\n\n=== Test {test_id} comparison ===")
    if isinstance(actual, pd.DataFrame) and isinstance(expected, pd.DataFrame):
        print(f"Shape: {actual.shape}")
        print("Actual preview:")
        print(actual.iloc[:3, :6].to_string(index=False))
        print("Expected preview:")
        print(expected.iloc[:3, :6].to_string(index=False))
        numeric_columns = [
            column
            for column in actual.columns.intersection(expected.columns)
            if pd.api.types.is_numeric_dtype(actual[column])
            and pd.api.types.is_numeric_dtype(expected[column])
        ]
        actual_values = actual[numeric_columns].to_numpy(dtype=float)
        expected_values = expected[numeric_columns].to_numpy(dtype=float)
    else:
        actual_values = np.asarray(actual, dtype=float)
        expected_values = np.asarray(expected, dtype=float)
        print(f"Shape: {actual_values.shape}")
        print("Actual:")
        print(_preview_array(actual_values))
        print("Expected:")
        print(_preview_array(expected_values))

    difference = actual_values - expected_values
    print("Difference (actual - expected):")
    print(_preview_array(difference))
    maximum_absolute_difference = float(np.max(np.abs(difference)))
    expected_scale = float(np.max(np.abs(expected_values)))
    print(f"Maximum absolute difference: {maximum_absolute_difference:.12g}")
    print(f"Mean absolute difference: {np.mean(np.abs(difference)):.12g}")
    if expected_scale > 0.0:
        print(
            "Maximum difference / largest expected magnitude: "
            f"{100.0 * maximum_absolute_difference / expected_scale:.6g}%"
        )


def assert_numeric_close(test_id: str, actual, expected, **kwargs) -> None:
    report_comparison(test_id, actual, expected)
    np.testing.assert_allclose(actual, expected, **kwargs)


def assert_dataframes_close(
    test_id: str, actual: pd.DataFrame, expected: pd.DataFrame, **kwargs
) -> None:
    report_comparison(test_id, actual, expected)
    assert_frame_equal(actual, expected, **kwargs)


def pairwise_covariance_from_input() -> np.ndarray:
    """Build Test 3's covariance input by running Test 1's function."""

    return missing_covariance(matrix("test1.csv"), skip_missing=False)


def pairwise_correlation_from_input() -> np.ndarray:
    """Build Test 3's correlation input by running Test 1's function."""

    return missing_covariance(
        matrix("test1.csv"), skip_missing=False, correlation=True
    )


def assert_psd_repair_properties(
    test: unittest.TestCase, actual: np.ndarray, original: np.ndarray
) -> None:
    """Check the mathematical properties required of a PSD repair."""

    np.testing.assert_allclose(actual, actual.T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.diag(actual), np.diag(original), rtol=1e-10, atol=1e-12
    )
    test.assertGreaterEqual(float(np.linalg.eigvalsh(actual).min()), -1e-8)


def assert_simulation_covariance_close(
    test: unittest.TestCase,
    test_id: str,
    actual: np.ndarray,
    expected: np.ndarray,
    target: np.ndarray,
    n_samples: int,
) -> None:
    """Compare simulated covariances using their theoretical sampling error.

    For multivariate Normal observations, the sample covariance follows a
    Wishart distribution and

        Var(S_ij) = (Sigma_ij^2 + Sigma_ii * Sigma_jj) / (n - 1).

    The professor and Python outputs are independent Monte Carlo estimates, so
    their difference has twice that variance. A four-standard-error limit is
    applied in addition to (not instead of) the original 2.5% matrix-scale
    guardrail, so this check is not loosened to make the output pass.
    """

    report_comparison(test_id, actual, expected)

    covariance_variance = (
        target * target + np.outer(np.diag(target), np.diag(target))
    ) / (n_samples - 1)
    one_run_standard_error = np.sqrt(covariance_variance)
    difference_standard_error = np.sqrt(2.0 * covariance_variance)

    actual_z = float(
        np.max(np.abs(actual - target) / one_run_standard_error)
    )
    expected_z = float(
        np.max(np.abs(expected - target) / one_run_standard_error)
    )
    difference_z = float(
        np.max(np.abs(actual - expected) / difference_standard_error)
    )
    scale_difference = float(
        np.max(np.abs(actual - expected)) / np.max(np.abs(expected))
    )

    if SHOW_TEST_OUTPUTS:
        print("Monte Carlo sampling check (Normal/Wishart theory):")
        print(f"  Python actual vs theoretical covariance: {actual_z:.6g} standard errors")
        print(f"  Professor expected vs theoretical covariance: {expected_z:.6g} standard errors")
        print(f"  Actual vs expected: {difference_z:.6g} combined standard errors")
        print(f"  Allowed limit: {SIMULATION_SIGMA_LIMIT:g} standard errors")
        print(
            "  Original matrix-scale guardrail: "
            f"{100.0 * scale_difference:.6g}% <= "
            f"{100.0 * SIMULATION_SCALE_LIMIT:g}%"
        )

    test.assertLessEqual(
        actual_z,
        SIMULATION_SIGMA_LIMIT,
        msg=(
            f"Test {test_id} Python covariance is {actual_z:.3f} standard "
            "errors from the theoretical covariance"
        ),
    )
    test.assertLessEqual(
        expected_z,
        SIMULATION_SIGMA_LIMIT,
        msg=(
            f"Test {test_id} professor covariance is {expected_z:.3f} standard "
            "errors from the theoretical covariance"
        ),
    )
    test.assertLessEqual(
        difference_z,
        SIMULATION_SIGMA_LIMIT,
        msg=(
            f"Test {test_id} actual/expected difference is {difference_z:.3f} "
            "combined standard errors"
        ),
    )
    test.assertLessEqual(
        scale_difference,
        SIMULATION_SCALE_LIMIT,
        msg=(
            f"Test {test_id} maximum covariance difference is "
            f"{100.0 * scale_difference:.3f}% of matrix scale, above the "
            f"original {100.0 * SIMULATION_SCALE_LIMIT:.1f}% guardrail"
        ),
    )


def pca_covariance_target(
    covariance: np.ndarray, explained_variance: float
) -> np.ndarray:
    """Return the covariance implied by retaining the requested PCA factors."""

    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues >= 1e-8
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    retained = int(
        np.searchsorted(
            np.cumsum(eigenvalues) / np.trace(covariance), explained_variance
        )
        + 1
    )
    root = eigenvectors[:, :retained] @ np.diag(np.sqrt(eigenvalues[:retained]))
    return root @ root.T


class TestMissingCovariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = matrix("test1.csv")

    def test_1_1_skip_missing_covariance(self) -> None:
        actual = missing_covariance(self.values, skip_missing=True)
        assert_numeric_close(
            "1.1", actual, matrix("testout_1.1.csv"), rtol=1e-10, atol=1e-12
        )

    def test_1_2_skip_missing_correlation(self) -> None:
        actual = missing_covariance(
            self.values, skip_missing=True, correlation=True
        )
        assert_numeric_close(
            "1.2", actual, matrix("testout_1.2.csv"), rtol=1e-10, atol=1e-12
        )

    def test_1_3_pairwise_covariance(self) -> None:
        actual = missing_covariance(self.values, skip_missing=False)
        assert_numeric_close(
            "1.3", actual, matrix("testout_1.3.csv"), rtol=1e-10, atol=1e-12
        )

    def test_1_4_pairwise_correlation(self) -> None:
        actual = missing_covariance(
            self.values, skip_missing=False, correlation=True
        )
        assert_numeric_close(
            "1.4", actual, matrix("testout_1.4.csv"), rtol=1e-10, atol=1e-12
        )


class TestExponentiallyWeightedCovariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = matrix("test2.csv")

    def test_2_1_ew_covariance_lambda_097(self) -> None:
        actual = ew_covariance(self.values, 0.97)
        assert_numeric_close(
            "2.1", actual, matrix("testout_2.1.csv"), rtol=1e-10, atol=1e-12
        )

    def test_2_2_ew_correlation_lambda_094(self) -> None:
        covariance = ew_covariance(self.values, 0.94)
        volatility = np.sqrt(np.diag(covariance))
        actual = covariance / np.outer(volatility, volatility)
        assert_numeric_close(
            "2.2", actual, matrix("testout_2.2.csv"), rtol=1e-10, atol=1e-12
        )

    def test_2_3_mixed_decay_covariance(self) -> None:
        actual = covariance_with_ew_variance(
            self.values, variance_decay=0.97, correlation_decay=0.94
        )
        assert_numeric_close(
            "2.3", actual, matrix("testout_2.3.csv"), rtol=1e-10, atol=1e-12
        )


class TestPsdRepair(unittest.TestCase):
    def test_3_1_near_psd_covariance(self) -> None:
        original = pairwise_covariance_from_input()
        actual = near_psd(original)
        assert_numeric_close(
            "3.1", actual, matrix("testout_3.1.csv"), rtol=1e-8, atol=1e-10
        )
        assert_psd_repair_properties(self, actual, original)

    def test_3_2_near_psd_correlation(self) -> None:
        original = pairwise_correlation_from_input()
        actual = near_psd(original)
        assert_numeric_close(
            "3.2", actual, matrix("testout_3.2.csv"), rtol=1e-8, atol=1e-10
        )
        assert_psd_repair_properties(self, actual, original)

    def test_3_3_higham_covariance(self) -> None:
        original = pairwise_covariance_from_input()
        actual = higham_nearest_psd(original)
        assert_numeric_close(
            "3.3", actual, matrix("testout_3.3.csv"), rtol=1e-7, atol=1e-9
        )
        assert_psd_repair_properties(self, actual, original)

    def test_3_4_higham_correlation(self) -> None:
        original = pairwise_correlation_from_input()
        actual = higham_nearest_psd(original)
        assert_numeric_close(
            "3.4", actual, matrix("testout_3.4.csv"), rtol=1e-7, atol=1e-9
        )
        assert_psd_repair_properties(self, actual, original)


class TestPsdCholesky(unittest.TestCase):
    def test_4_1_chol_psd(self) -> None:
        psd_input = near_psd(pairwise_covariance_from_input())
        actual = chol_psd(psd_input)
        assert_numeric_close(
            "4.1", actual, matrix("testout_4.1.csv"), rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(actual @ actual.T, psd_input, atol=1e-8)


class TestNormalSimulation(unittest.TestCase):
    def _compare_normal(
        self,
        test_id: str,
        input_name: str,
        output_name: str,
        fix_method=near_psd,
        repair_target: bool = False,
    ) -> None:
        covariance = matrix(input_name)
        samples = simulate_normal(
            100_000, covariance, seed=1234, fix_method=fix_method
        )
        actual = np.cov(samples, rowvar=False, ddof=1)
        target = fix_method(covariance) if repair_target else covariance
        assert_simulation_covariance_close(
            self,
            test_id,
            actual,
            matrix(output_name),
            target,
            samples.shape[0],
        )

    def test_5_1_positive_definite_input(self) -> None:
        self._compare_normal("5.1", "test5_1.csv", "testout_5.1.csv")

    def test_5_2_positive_semidefinite_input(self) -> None:
        self._compare_normal("5.2", "test5_2.csv", "testout_5.2.csv")

    def test_5_3_non_psd_near_psd_fix(self) -> None:
        self._compare_normal(
            "5.3", "test5_3.csv", "testout_5.3.csv", near_psd, True
        )

    def test_5_4_non_psd_higham_fix(self) -> None:
        self._compare_normal(
            "5.4",
            "test5_3.csv",
            "testout_5.4.csv",
            higham_nearest_psd,
            True,
        )

    def test_5_5_pca_99_percent(self) -> None:
        covariance = matrix("test5_2.csv")
        samples = simulate_pca(
            covariance, 100_000, explained_variance=0.99, seed=1234
        )
        actual = np.cov(samples, rowvar=False, ddof=1)
        target = pca_covariance_target(covariance, 0.99)
        assert_simulation_covariance_close(
            self,
            "5.5",
            actual,
            matrix("testout_5.5.csv"),
            target,
            samples.shape[0],
        )


class TestReturns(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prices = pd.read_csv(DATA / "test6.csv")

    def test_6_1_arithmetic_returns(self) -> None:
        actual = calculate_returns(self.prices, method="DISCRETE", date_column="Date")
        expected = pd.read_csv(DATA / "testout6_1.csv")
        assert_dataframes_close(
            "6.1",
            actual,
            expected,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_6_2_log_returns(self) -> None:
        actual = calculate_returns(self.prices, method="LOG", date_column="Date")
        expected = pd.read_csv(DATA / "testout6_2.csv")
        assert_dataframes_close(
            "6.2",
            actual,
            expected,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )


class TestDistributionFitting(unittest.TestCase):
    def test_7_1_fit_normal(self) -> None:
        values = matrix("test7_1.csv")[:, 0]
        expected = pd.read_csv(DATA / "testout7_1.csv").iloc[0]
        fit = fit_normal(values)
        assert_numeric_close(
            "7.1",
            [fit.mu, fit.sigma],
            expected[["mu", "sigma"]].to_numpy(float),
            rtol=1e-10,
        )

    def test_7_2_fit_student_t(self) -> None:
        values = matrix("test7_2.csv")[:, 0]
        expected = pd.read_csv(DATA / "testout7_2.csv").iloc[0]
        fit = fit_student_t(values)
        assert_numeric_close(
            "7.2",
            [fit.mu, fit.sigma, fit.nu],
            expected[["mu", "sigma", "nu"]].to_numpy(float),
            rtol=2e-5,
            atol=1e-8,
        )

    def test_7_3_t_regression(self) -> None:
        data = pd.read_csv(DATA / "test7_3.csv")
        expected = pd.read_csv(DATA / "testout7_3.csv").iloc[0]
        fit = fit_t_regression(data["y"], data.drop(columns="y"))
        actual = np.concatenate(
            [[fit.mu, fit.sigma, fit.nu, fit.alpha], fit.beta]
        )
        target = expected[["mu", "sigma", "nu", "Alpha", "B1", "B2", "B3"]].to_numpy(float)
        assert_numeric_close("7.3", actual, target, rtol=2e-5, atol=1e-7)

    def test_7_4_aicc_on_fitted_t(self) -> None:
        values = matrix("test7_2.csv")[:, 0]
        expected = float(pd.read_csv(DATA / "testout7_4.csv").iloc[0, 0])
        actual = aicc(fit_student_t(values), values)
        report_comparison("7.4", actual, expected)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_7_5_fit_nig_by_moments(self) -> None:
        values = matrix("test7_5.csv")[:, 0]
        expected = pd.read_csv(DATA / "testout7_5.csv").iloc[0]
        fit = fit_nig_moments(values)
        assert_numeric_close(
            "7.5",
            [fit.mu, fit.alpha, fit.beta, fit.delta],
            expected[["mu", "alpha", "beta", "delta"]].to_numpy(float),
            rtol=1e-10,
            atol=1e-12,
        )

        # The method-of-moments parameters reproduce the four sample moments.
        sample_moments = [
            np.mean(values),
            np.var(values, ddof=1),
            stats.skew(values, bias=True),
            stats.kurtosis(values, fisher=True, bias=True),
        ]
        fitted_moments = [
            fit.mean,
            fit.variance,
            fit.skewness,
            fit.excess_kurtosis,
        ]
        if SHOW_TEST_OUTPUTS:
            print("NIG method-of-moments objective:")
            for name, sample, fitted in zip(
                ["mean", "variance", "skewness", "excess kurtosis"],
                sample_moments,
                fitted_moments,
            ):
                print(
                    f"  {name}: sample={sample:.12g}, "
                    f"fitted={fitted:.12g}, difference={fitted - sample:.3g}"
                )
        np.testing.assert_allclose(fitted_moments, sample_moments, rtol=1e-10)

    def test_7_6_fit_nig_by_mle(self) -> None:
        values = matrix("test7_5.csv")[:, 0]
        expected = pd.read_csv(DATA / "testout7_6.csv").iloc[0]
        fit = fit_nig_mle(values)
        assert_numeric_close(
            "7.6",
            [fit.mu, fit.alpha, fit.beta, fit.delta],
            expected[["mu", "alpha", "beta", "delta"]].to_numpy(float),
            rtol=1e-8,
            atol=1e-10,
        )

        # MLE optimizes likelihood, so it should beat the moment-matched fit.
        moments_fit = fit_nig_moments(values)
        mle_log_likelihood = fit.log_likelihood(values)
        moments_log_likelihood = moments_fit.log_likelihood(values)
        if SHOW_TEST_OUTPUTS:
            print("NIG maximum-likelihood objective:")
            print(f"  MLE log likelihood: {mle_log_likelihood:.12g}")
            print(f"  Moments-fit log likelihood: {moments_log_likelihood:.12g}")
            print(
                "  MLE improvement: "
                f"{mle_log_likelihood - moments_log_likelihood:.12g}"
            )
        self.assertGreaterEqual(
            mle_log_likelihood, moments_log_likelihood
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
