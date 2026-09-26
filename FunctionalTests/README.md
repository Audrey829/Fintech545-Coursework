# Functional Tests 1.1–13.9

This folder turns the professor's official input/output CSV files into 50
repeatable functional tests for the reusable `risk_management` Python package.

## What is actually being tested

The data flow is always:

`professor input CSV → our risk_management function → actual result → compare with professor testout`

The calculation functions never read `testout*.csv`. Expected outputs are read
only by the test files after an actual result has been calculated. This avoids
the invalid pattern of returning a provided answer instead of computing it.

- `test_tests_1_to_7.py` contains Tests 1.1–7.6.
- `test_tests_8_to_13.py` contains Tests 8.1–13.9.
- `run_tests_with_outputs.py` discovers both files and preserves numeric test
  order in its terminal report.

## Run all 50 tests

From the `Fintech545-Coursework` repository root:

```bash
source .venv/bin/activate
python -m unittest discover -s FunctionalTests -p 'test_*.py' -v
```

Test 12.3 reproduces a 500-step American option tree with two discrete cash
dividends, so a complete run can take about one to two minutes on a laptop.

## Directly view actual output and differences

To see every calculated result beside the professor's expected result:

```bash
python FunctionalTests/run_tests_with_outputs.py
```

Detailed mode prints:

- `Actual`: calculated from the professor input by our function.
- `Expected`: read from the matching professor `testout` file.
- `Difference (actual - expected)`.
- Maximum and mean absolute differences.
- Monte Carlo standard errors for simulation-based tail-risk tests.

Small results are printed in full. Large matrices and tables show a preview
plus overall difference statistics. Assertions are unchanged in detailed mode,
so final `OK` still means every displayed difference passed its stated check.

For a new environment:

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
| 6.1–6.2 | `calculate_returns` | Arithmetic and logarithmic returns |
| 7.1–7.4 | distribution fitting and `aicc` | Normal/t fits and t-error regression |
| 7.5–7.6 | `fit_nig_moments`, `fit_nig_mle` | NIG moment matching and maximum likelihood |
| 8.1–8.6 | `value_at_risk`, `expected_shortfall` | Parametric and simulated VaR/ES |
| 9.1 | copula simulation, `aggregate_portfolio_risk` | Gaussian-copula portfolio risk |
| 10.1–10.4 | `risk_parity`, `maximize_sharpe_ratio` | Portfolio optimization |
| 11.1–11.2 | `expost_factor` | Return and volatility attribution |
| 12.1–12.3 | `gbsm`, American option functions | European/American values and Greeks |
| 13.1–13.4 | `kendall_correlation`, `fit_multivariate_t` | Multivariate-t estimation |
| 13.5–13.8 | Gaussian/t copula functions | Likelihood, selection, tail dependence |
| 13.9 | t-copula simulation, `aggregate_portfolio_risk` | t-copula portfolio risk |

The matching source-file map and direct-call examples are also available in
[`risk_management/README.md`](../risk_management/README.md).

## Validation rules

Deterministic calculations use tight numerical tolerances and also check
mathematical properties where useful (PSD matrices, fully invested portfolios,
option Greeks, attribution totals, and information-criterion ordering).

Julia and NumPy do not generate the same random stream from the same numeric
seed. Tests 5.1–5.5 therefore use Normal/Wishart covariance standard errors.
Tests 8.3, 8.6, 9.1, and 13.9 use asymptotic standard errors for empirical VaR
and ES, requiring actual-versus-expected differences to remain within four
combined standard errors. They also retain a separate relative guardrail: 5%
at 100,000 draws, scaled at the theoretical `1/sqrt(n)` rate for smaller runs.
These checks test whether two independent Monte Carlo estimates are
scientifically consistent; they do not replace or weaken the underlying model.

## Quick calls example

These are direct reusable calls, without running the functional-test harness:

```python
import pandas as pd
from risk_management import (
    aggregate_portfolio_risk,
    calculate_returns,
    expected_shortfall,
    fit_student_t,
    higham_nearest_psd,
    risk_parity,
    simulate_normal,
    value_at_risk,
)

prices = pd.read_csv("prices.csv")
returns = calculate_returns(prices, method="LOG", date_column="Date")

fit = fit_student_t(returns["Asset1"])
print(value_at_risk(fit), expected_shortfall(fit))

covariance = higham_nearest_psd(returns.drop(columns="Date").cov())
weights = risk_parity(covariance)
scenarios = simulate_normal(100_000, covariance, seed=545)

current_values = [2_000, 3_000]
risk = aggregate_portfolio_risk(
    scenarios[:, :2], current_values, ["Asset1", "Asset2"]
)
print(weights)
print(risk)
```

Course-name aliases such as `ewCovar`, `simulateNormal`, `fit_general_t`, and
`return_calculate` remain exported for easy comparison with the Julia library.

## References

- Professor library: <https://github.com/dompazz/FinTech-545-Fall2026/tree/main/library>
- Professor functional-test data: <https://github.com/dompazz/FinTech-545-Fall2026/tree/main/testfiles/data>
