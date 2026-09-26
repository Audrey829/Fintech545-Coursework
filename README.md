# Fintech545-Coursework

Coursework and reusable Python tools for FinTech 545 Quantitative Risk
Management.

## Repository overview

- `Assignment1/` contains the Assignment 1 notebooks, data, and written work.
- `risk_management/` contains reusable covariance, simulation, distribution,
  tail-risk, portfolio, attribution, option, multivariate-t, and copula tools.
- `FunctionalTests/` contains the professor's official inputs and expected
  outputs together with the automated functional tests.

## Functional tests

The current suite covers Tests 1.1–13.9: 50 tests that calculate each result
through `risk_management` before comparing it with the professor's expected
output.

Run the complete suite from the repository root:

```bash
source .venv/bin/activate
python -m unittest discover -s FunctionalTests -p 'test_*.py' -v
```

To view actual, expected, and actual-minus-expected values directly:

```bash
python FunctionalTests/run_tests_with_outputs.py
```

See [FunctionalTests/README.md](FunctionalTests/README.md) for the full test
map, validation rules, setup, and direct midterm/final calls. See
[risk_management/README.md](risk_management/README.md) for the function-module
map. Assignment instructions remain in
[Assignment1/README.md](Assignment1/README.md).
