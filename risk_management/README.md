# `risk_management` function map

This package stores the reusable implementations called by the functional
tests. It does not read the professor's `testout*.csv` files; only the test
files read those expected-output fixtures.

| File | Functional tests | Main functions |
| --- | --- | --- |
| `covariance.py` | 1.1–2.3 | `missing_covariance`, `ew_covariance`, `covariance_with_ew_variance` |
| `psd.py` | 3.1–4.1 | `near_psd`, `higham_nearest_psd`, `chol_psd` |
| `simulation.py` | 5.1–5.5 | `simulate_normal`, `simulate_pca` |
| `returns.py` | 6.1–6.2 | `calculate_returns` |
| `distributions.py` | 7.1–7.6 | Normal, Student-t, regression-t, and NIG fitting; `aicc` |
| `risk_metrics.py` | 8.1–9.1, 13.9 | `value_at_risk`, `expected_shortfall`, `aggregate_portfolio_risk` |
| `portfolio.py` | 10.1–10.4 | `risk_parity`, `maximize_sharpe_ratio` |
| `attribution.py` | 11.1–11.2 | `expost_factor` |
| `options.py` | 12.1–12.3 | `gbsm`, continuous/discrete-dividend American option functions |
| `multivariate.py` | 9.1, 13.1–13.9 | Kendall correlation, multivariate-t, Gaussian/t copulas |

## Quick direct calls

```python
from risk_management import (
    expected_shortfall,
    fit_multivariate_t,
    fit_student_t,
    gbsm,
    maximize_sharpe_ratio,
    risk_parity,
    value_at_risk,
)

t_fit = fit_student_t(asset_returns)
var95 = value_at_risk(t_fit)
es95 = expected_shortfall(t_fit)

risk_parity_weights = risk_parity(covariance)
max_sharpe_weights = maximize_sharpe_ratio(covariance, means, 0.04)

european_call = gbsm(
    True, 100, 100, 50 / 365, 0.045, 0.045, 0.20,
    include_greeks=True,
)

multivariate_t = fit_multivariate_t(return_matrix)
```

All public functions are exported from `risk_management/__init__.py`, whose
module-level annotation contains the same test-to-file map.
