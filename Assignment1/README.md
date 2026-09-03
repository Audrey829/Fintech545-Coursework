# FinTech 545 — Assignment 1

This folder contains the data, calculations, figures, and written responses for Assignment 1. Run the five problem notebooks to reproduce the numerical results and plots reported in `assignment_1.pdf`.

## Files

- ` assignment_1.pdf` — written responses for Problems 1–5
- `problem1_code.ipynb` — sample moments and fitted-Normal tail comparison
- `problem2_code.ipynb` — scatter plot; OLS, Normal-error MLE, and Student-t-error MLE; AICc and error-quantile comparisons
- `problem3_code.ipynb` — pairwise scatter plots and Pearson/Spearman correlation comparisons
- `problem4_code.ipynb` — covariance matrix, conditional mean and variance, 95% band, and coverage calculations
- `problem5_code.ipynb` — time-series, ACF, and PACF plots; AR/MA fits and AICc comparison
- `problem1.csv` through `problem5.csv` — input data for the corresponding problem


## Environment

Python 3 with Jupyter is required. Install the packages used by the notebooks with:

```bash
python3 -m pip install jupyter numpy pandas scipy matplotlib statsmodels
```

## Reproduce the results

Clone the repository, enter the assignment directory, and start Jupyter Lab:

```bash
git clone https://github.com/Audrey829/Fintech545-Coursework.git
cd Fintech545-Coursework/Assignment1
jupyter lab
```

In Jupyter Lab, open each code notebook and use **Run → Run All Cells** in this order:

1. `problem1_code.ipynb`
2. `problem2_code.ipynb`
3. `problem3_code.ipynb`
4. `problem4_code.ipynb`
5. `problem5_code.ipynb`

Run the notebooks from the `Assignment1` directory because each notebook reads its corresponding CSV file using a relative path such as `problem1.csv`. No external data or command-line arguments are required.

The notebooks print the reported statistics, parameter estimates, likelihoods, AICc values, quantiles, correlations, conditional-coverage results, and AR/MA model comparison. Figures are displayed inline. `assignment_1.pdf` contains the written interpretation of those results.
