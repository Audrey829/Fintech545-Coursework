# Functional Tests 1.1–7.6

This folder turns the professor's input/output CSV files into repeatable tests
for the reusable `risk_management` Python package in the repository root.

## Run all 25 tests

From the `Fintech545-Coursework` repository root:

```bash
source .venv/bin/activate
python -m unittest discover -s FunctionalTests -p 'test_*.py' -v
```

## Directly view actual output and differences

To see each function's calculated output beside the professor's expected
output, run:

```bash
python FunctionalTests/run_tests_with_outputs.py
```

For every test, detailed mode prints:

- `Actual`: the result calculated by our function from the input data.
- `Expected`: the professor's matching `testout` data.
- `Difference (actual - expected)`.
- Maximum absolute difference and mean absolute difference.

Small results are printed in full. Large matrices and returns tables show a
preview plus the overall difference statistics so the terminal stays readable.
The tests still perform the same assertions, so the final `OK` means every
displayed difference is within that test's allowed tolerance.

If the environment is new, install the three runtime dependencies first:

```bash
python -m pip install -r FunctionalTests/requirements.txt
```

## Test map

| Tests | Reusable functions | Topic |
| --- | --- | --- |
| 1.1–1.4 | `missing_covariance` | Complete-case and pairwise covariance/correlation |
| 2.1–2.3 | `ew_covariance`, `covariance_with_ew_variance` | Exponentially weighted covariance |
| 3.1–3.4 | `near_psd`, `higham_nearest_psd` | Repairing non-PSD matrices |
| 4.1 | `chol_psd` | PSD Cholesky factorization |
| 5.1–5.5 | `simulate_normal`, `simulate_pca` | Multivariate simulation |
| 6.1–6.2 | `calculate_returns` | Arithmetic and log returns |
| 7.1–7.4 | distribution fitting and `aicc` | Normal/t fits and t-error regression |
| 7.5–7.6 | `fit_nig_moments`, `fit_nig_mle` | NIG moment matching and maximum likelihood |

The deterministic calculations use tight equality tolerances. Julia and NumPy
generate different random streams even when they use the same seed, so Tests
5.1–5.5 cannot require identical sample covariance matrices. Those tests use
the theoretical sampling standard error of a Normal sample covariance,

`SE(S_ij) = sqrt((Sigma_ij^2 + Sigma_ii * Sigma_jj) / (n - 1))`,

and require the Python result, professor result, and their combined difference
to stay within four standard errors. The original 2.5% matrix-scale ceiling is
also retained as an additional guardrail, so the new validation is stricter,
not a relaxed way to obtain `OK`.

## Quick Test Examples

```python
import pandas as pd
from risk_management import (
    calculate_returns,
    ew_covariance,
    fit_student_t,
    fit_nig_mle,
    fit_nig_moments,
    higham_nearest_psd,
    simulate_normal,
)

prices = pd.read_csv("prices.csv")
returns = calculate_returns(prices, method="LOG", date_column="Date")

covariance = ew_covariance(returns.drop(columns="Date"), 0.97)
covariance = higham_nearest_psd(covariance)
scenarios = simulate_normal(100_000, covariance, seed=545)

asset_fit = fit_student_t(returns["Asset1"])
print(asset_fit.mu, asset_fit.sigma, asset_fit.nu)

nig_moments = fit_nig_moments(returns["Asset1"])
nig_mle = fit_nig_mle(returns["Asset1"])
print(nig_moments)
print(nig_mle)
```

The two NIG fits generally have different parameters. Method of moments forces
the fitted distribution to reproduce the sample mean, variance, skewness, and
excess kurtosis. MLE instead chooses parameters that maximize the likelihood of
all observations. With a finite sample, sampling noise and tail observations
make those two objectives disagree; the MLE fit should have the higher sample
log likelihood, while the moment fit should match the four sample moments.

Course-name aliases such as `ewCovar`, `simulateNormal`, `fit_general_t`, and
`return_calculate` are also exported for easy comparison with the Julia files.

## References

- Professor library: <https://github.com/dompazz/FinTech-545-Fall2026/tree/main/library>
- Professor functional-test data: <https://github.com/dompazz/FinTech-545-Fall2026/tree/main/testfiles/data>
