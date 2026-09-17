# Fintech545-Coursework

Coursework and reusable Python tools for FinTech 545 Quantitative Risk
Management.

## Repository overview

- `Assignment1/` contains the Assignment 1 notebooks, data, and written work.
- `risk_management/` contains reusable covariance, simulation, return, matrix,
  and distribution-fitting functions.
- `FunctionalTests/` contains the professor's supplied inputs and expected
  outputs together with the automated functional tests.

## Functional tests

The current suite covers Tests 1.1–7.6. It runs 25 tests that calculate each
result through the `risk_management` package and then compare it with the
professor's expected output.

Run the complete suite from the repository root:

```bash
source .venv/bin/activate
python -m unittest discover -s FunctionalTests -p 'test_*.py' -v
```

See [FunctionalTests/README.md](FunctionalTests/README.md) for the test map,
setup instructions, NIG fitting details, and quick tests. See
[Assignment1/README.md](Assignment1/README.md) for Assignment 1 instructions.
