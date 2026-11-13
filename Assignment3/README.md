# Assignment 3 — FinTech 545

## Open the answer

Open `assignment3.ipynb` directly in JupyterLab or VS Code. It contains the written
answers, function calls, saved numeric tables, and four figures. Each problem
has classroom reasoning, Predict, Fit, Reconcile, and an observed-results
interpretation. `assignment3_answer.pdf` contains the pdf version of `assignment3.ipynb`.


## Folder layout

Keep `Assignment3/` beside `risk_management/`:

    coursework-root/
      risk_management/                 existing course library + extension
        coursework_extensions.py       new reusable functions
      Assignment3/
        assignment3.ipynb               complete executed notebook
        assignment3.py                 equivalent cell-marked source
        run_notebook.py                 standalone sequential execution helper
        requirements.txt
        README.md
        Assignment 3.pdf               instructor prompt
        problem1.csv
        problem3.csv
        problem3_portfolios.csv
        problem4.csv
        problem5.csv
        F-F_Research_Data_Factors_daily.CSV
        F-F_Momentum_Factor_daily.CSV
        figure_1.png ... figure_4.png
        results.json                   machine-readable verification results

## Reproduce

Python 3.10+ recommended. From coursework-root:

```bash
python -m pip install -r Assignment3/requirements.txt
python -m jupyter lab
```

Open `Assignment3/assignment3.ipynb`, select this Python environment, and choose
Restart Kernel and Run All Cells. To create a second executed copy from a terminal:

```bash
python -m jupyter nbconvert --to notebook --execute Assignment3/assignment3.ipynb \
  --output assignment3_rerun.ipynb --ExecutePreprocessor.timeout=600
```

Alternatively, from the repository root or from Assignment3, execute the script:

```bash
python Assignment3/assignment3.py
```

(The above relative script path assumes the repository root.)

A standalone runner is included for environments unable to launch a Jupyter
kernel or initialize IPython. It executes cells in one shared Python namespace,
captures display tables and figures, and updates the saved notebook outputs:

```bash
MPLCONFIGDIR=/tmp/fintech545-mpl python Assignment3/run_notebook.py Assignment3/assignment3.ipynb
```

This runner replaces only the IPython display interface during its own process;
it does not install or change any package. Its stdout messages are grouped at
the end of each cell; tables and plots preserve display order. It is not a live
Jupyter kernel. The notebook itself uses normal IPython display when opened
in Jupyter.

## Existing library use and extensions

The provided `risk_management` package is a copy of the user's existing local
coursework library. All pre-existing Python sources are unchanged. In particular:

- `options.gbsm`: scalar GBSM and analytic Greeks.
- `options.american_continuous` / `american_discrete_dividends`: American trees.
- `returns.calculate_returns`: arithmetic returns.
- `statistics.sample_moments`: bias-corrected skewness/excess kurtosis.
- `distributions.fit_normal`, `fit_student_t`, `aicc`: MLE/model comparison.
- `risk_metrics.value_at_risk`, `expected_shortfall`, `simulate_fitted_distribution`.
- `portfolio.maximize_sharpe_ratio`: constrained maximum-Sharpe optimization.

New generic diagnostics/orchestration live in `coursework_extensions.py`, not
inside notebook cells. They provide bid/ask/mid IV inversion with arbitrage bounds,
parity and IV-gap checks, a European two-dividend non-recombining tree, tree
convergence and sensitivity calls, vectorized GBSM scenario valuation, book
revaluation and Taylor decomposition, full/downside CAPM reporting, efficient
frontiers, factor-model reconciliation, and bootstrap diagnostics. Numerical
root finding and constrained optimization use SciPy; OLS uses statsmodels.
Option prices use our course GBSM/tree code, not a third-party option package.

## Conventions that affect reproduction

- P1: 35 calendar days/365; carry = r − q; parity tolerance is the **sum of full spreads**.
- P2: N=200 for reported sensitivities; all dividend dates align exactly with nodes.
  The convergence table includes N=50/55,100/105,150/155,200/205,300.
  No-dividend and dividend EEPs subtract European and American values from the
  same N=200 tree; raw tree-minus-GBSM error is reported separately.
- P3: 255 days/year; ten compounded daily arithmetic draws per path; 200,000 paths
  per model; seeds 545 and 546; zero fitted locations for simulation; constant IV.
  All holdings are single-share units. No unstated cash financing is added.
  Normal delta risk uses the Normal MLE sigma. Taylor comparison includes gamma
  **and** calendar theta. Returns <= −100% are rejected, never silently clipped.
- P4: monthly rf=0.03/12; annual arithmetic mean=12×monthly mean; annual volatility
  =√12×monthly standard deviation. Historical VaR/ES use the original course
  order-statistic convention, including all observations below its VaR cutoff.
  Normal estimates use sample standard deviation. Bootstrap: 200 replications,
  seed 545, joint row resampling, long-only constraints.
- P5: factors divided by 100 exactly once and joined by date. Annual arithmetic
  expectations and Sharpe ratios use 252 days/year; CAGR is also shown but is a
  different statistic.
  The latest ten-calendar-year factor window is 2014-10-01 to 2024-09-30.
  OLS SEs are conventional, not HAC. AICc includes residual variance in k.
  The momentum stress isolates factor exposure, omitting alpha/RF accrual.

## Validation completed

All 17 code cells executed sequentially without errors through a real Jupyter
kernel using the course virtual environment. Saved notebook structure was
validated with nbformat, and all four saved figures were visually inspected.
Full HTML rendering was not used for validation because the local Anaconda
nbconvert process exited abnormally; this does not affect notebook execution.
No answer PDF was generated or tested.

Checks passed for vector/scalar GBSM agreement, original/extension American tree
agreement, European discrete-dividend put-call parity, American >= European
values on the same dividend grid, IV repricing, portfolio constraints, exact
fund/factor date coverage, and omitted-factor alpha reconciliation. The 50
pre-existing FunctionalTests covering tests 1 through 13 also passed.

The simulation figures are point estimates from fixed seeds; tail Monte Carlo
uncertainty and finite-N tree error remain. With 240 monthly observations, 1%
historical ES is based on only about two or three tail months. These limits are
explained in the written answers.

## Source provenance

Instructor inputs are copied without edits from
`dompazz/FinTech-545-Fall2026`, commit
`5958d11b51960c18be98f6df2b97c4e80670fb2d`, directory
`Assignments/Assignment3`. `source_manifest.json` records SHA-256 hashes for
inputs and pre-existing library sources.
