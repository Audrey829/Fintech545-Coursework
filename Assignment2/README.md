# Assignment 2

This folder contains the complete Assignment 2 submission for FinTech 545.

## Files

- `Homework2.ipynb` — executable solution notebook with all predictions, fitted results, reconciliations, tables, and figures.
- `Homework2.html` — self-contained rendered notebook with code, outputs, and written answers visible.
- `Homework2_Solutions.pdf` — written answer exported from the executed notebook. This is the answer PDF.
- `Assignment 2.pdf` — professor's original problem statement only; it is not the answer PDF.
- `assignment.qmd` — professor's source instructions.
- `problem1.csv` through `problem5.csv` — professor-provided data.
- `risk_management/` — self-contained submission copy of the reusable local course library.
- `../risk_management/` — canonical repository-level source of that package. The notebook imports functions exported by `risk_management`; their implementations live in thematic modules, and there is no Assignment-2-specific calculation module.
- `requirements.txt` — Python dependencies.

The source files were copied from `dompazz/FinTech-545-Fall2026`, directory `Assignments/Assignment2`, at commit `6f104966e891a3326898d51e291c8a0153b454f7` (2026-10-01).

## Reproduce every result

Use Python **3.9.6** and open a terminal at the cloned repository root (whatever
name you gave that folder):

```bash
# Run the next line only if .venv does not already exist.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r Assignment2/requirements.txt
python -m jupyter nbconvert \
  --execute \
  --to notebook \
  Assignment2/Homework2.ipynb \
  --output-dir Assignment2 \
  --output Homework2_rerun.ipynb \
  --ExecutePreprocessor.timeout=180 \
  --ExecutePreprocessor.kernel_name=python3
```

The notebook uses fixed random seeds and should reproduce the saved results. It locates the data whether it is run from the repository root, `Assignment1`, or `Assignment2`.

To work interactively:

```bash
source .venv/bin/activate
python -m jupyter lab
```

Then open `Assignment2/Homework2.ipynb`. `Assignment1/Homework2.ipynb` is only a
relative symbolic link to that file for the supplied local Jupyter URL, not an
independent third copy.

## Library implementation notes

- Complete-case/pairwise covariance and EW covariance come from the implementations used by Functional Tests 1.1–2.3.
- Rebonato–Jäckel and Higham PSD repairs come from `risk_management/psd.py`, the implementations used by Functional Tests 3.1–3.4. They use NumPy eigendecompositions and SciPy matrix square roots.
- Normal and Student-t fitting come from `risk_management/distributions.py`, used by Functional Tests 7.1–7.4. Student-t MLE delegates to `scipy.stats.t.fit`.
- VaR and ES come from `risk_management/risk_metrics.py`, used by Functional Tests 8, 9, and 13.9.
- Kendall's τ, Gaussian/t copula likelihoods, simulation, and tail dependence come from `risk_management/multivariate.py`, used by Functional Tests 13.1–13.9. Kendall's τ delegates to `scipy.stats.kendalltau`; the likelihoods use SciPy multivariate densities.
- Reusable diagnostics that were not part of the original 50 tests were added to their natural modules (`covariance.py`, `distributions.py`, `risk_metrics.py`, `multivariate.py`, `statistics.py`, and `factor_models.py`). The root package is the canonical source; a synchronization test requires every Python source file in the self-contained submission copy to match it byte for byte.
- `simulate_market_model` uses zero-mean market and residual shocks and deliberately omits the fitted intercepts, as Problem 5 assumes zero expected returns.

From the repository root, rerun both the professor regression suite and the
extension suite with:

```bash
python -m unittest discover -s FunctionalTests -p 'test_*.py'
```

The extension tests directly cover: missing-data counts and correlations;
sample standard deviations; PSD repair diagnostics and eigenvector loadings;
return demeaning; moments, scale-mixture kurtosis, and outlier sensitivity;
threshold and joint-tail comparisons; fitted margins and copula fitting,
likelihood reconciliation, simulation, and risk differences; market-model
simulation with full versus diagonal residual covariance; EWMA diagnostics and
VaR sampling noise; scenario P&L; and synchronization of the two library copies.

## Validation

The submitted notebook was executed from top to bottom: 22 code cells completed
in order with no errors. The local library passed all 50 professor functional
tests plus 20 extension tests. The answer PDF begins
directly with Problem 1; the earlier title/conventions front matter has been
removed. The HTML and PDF were rendered from that executed notebook and visually
checked for readable tables, figures, headings, and pagination.
