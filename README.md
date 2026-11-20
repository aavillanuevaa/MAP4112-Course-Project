# MAP 4112 Course Project: Least Squares vs. Ridge vs. Lasso

Track A (real-data study). We compare ordinary least squares (OLS), ridge and
lasso regression for predicting MLB player salaries from batting statistics.

**Question.** The predictors are highly collinear. Does that make regularization
pay off, and does the payoff grow as the training set gets smaller?

## Data

**Hitters** (James et al., *An Introduction to Statistical Learning*), loaded
through the `ISLP` package, so there is nothing to download by hand.

- 322 players. The 59 with a missing salary are dropped, leaving **263 rows**.
- **19 predictors**: 16 numeric batting/fielding stats (1986 season and career)
  plus 3 one-hot league/division indicators.
- Target: **log(Salary)**. Raw salary is right-skewed (skew 1.59, which drops
  to −0.18 after the log transform).
- Strong collinearity: the career stats (CAtBat, CHits, CRuns, CRBI) correlate
  at 0.95–0.995, with VIFs from 130 to 500.

All features are standardized inside each method's pipeline, with the scaler
fit on the training split only.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data checks
```bash
python explore.py
```
Writes `results/eda/`: a text summary plus target, correlation and VIF plots.

### 2. Fit a method
```bash
python train.py --method ols
python train.py --method ridge
python train.py --method lasso
```
Optional: `--reps N` sets the number of random 70/30 splits (default 50). λ is
chosen by 10-fold cross-validation on each training split. Results are written
to `results/<method>/`.

### 3. Re-plot without refitting
```bash
python evaluate.py --method lasso
```

### 4. Compare methods
```bash
python compare.py
```
Reads every `results/*/metrics.json` and prints a table. Saves
`results/comparison.csv`, `comparison.png` and `coefficients_compare.png`.

### 5. Sample-size experiment
```bash
python sample_size.py
```
Refits every method at training sizes n = 40, 60, 100, 150 and 200, with 40
random draws each. Writes `results/sample_size/`.

## Project Structure

```
├── data.py              # load Hitters, clean, log-transform target, split
├── train.py             # fit one method over repeated splits, save results
├── evaluate.py          # regenerate a method's plots from saved CSVs
├── compare.py           # comparison table and plots across methods
├── explore.py           # data checks (missingness, skew, correlation, VIF)
├── sample_size.py       # test error vs. training size n
├── methods/
│   ├── __init__.py      # registry: name -> build_model
│   ├── ols.py
│   ├── ridge.py         # RidgeCV, 10-fold CV over λ ∈ [1e-2, 1e5]
│   └── lasso.py         # LassoCV, 10-fold CV over 100 λ values
└── results/
    ├── comparison.csv / comparison.png / coefficients_compare.png
    ├── eda/
    ├── sample_size/
    └── <method>/
        ├── metrics.json        # summary numbers
        ├── split_errors.csv    # per-split test MSE, λ, # nonzero coefs
        ├── coefficients.csv    # full-data standardized coefficients
        ├── path.csv            # coefficient path (ridge/lasso)
        ├── cv_curve.csv        # CV error vs λ (ridge/lasso)
        └── *.png
```
## Results

Test MSE on log(Salary) over 50 random 70/30 splits:

| Method | Test MSE (mean ± sd) | vs. OLS | Beats OLS on | Nonzero coefs |
|---|---|---|---|---|
| Ridge | 0.427 ± 0.069 | −4.6% | 74% of splits | 19 / 19 |
| Lasso | 0.431 ± 0.071 | −3.7% | 74% of splits | 10.9 / 19 |
| OLS   | 0.448 ± 0.083 | — | — | 19 / 19 |

Median test MSE by training size:

| n_train | OLS | Ridge | Lasso |
|---|---|---|---|
| 40  | 0.933 | 0.463 | 0.488 |
| 60  | 0.621 | 0.441 | 0.461 |
| 100 | 0.501 | 0.430 | 0.437 |
| 150 | 0.471 | 0.420 | 0.435 |
| 200 | 0.416 | 0.402 | 0.404 |

- **Collinearity makes OLS unstable.** OLS gives near-duplicate predictors large
  coefficients of opposite sign (AtBat −0.44 vs. Hits +0.59; CWalks −0.39 vs.
  CRuns +0.50). Ridge shrinks each correlated group together. Lasso keeps one
  member of each group (e.g. CHits) and zeros out the others (CAtBat, CRuns,
  CRBI).
- **The benefit grows as n shrinks.** At n = 40, regularization halves the test
  error. By n = 200, the three methods are nearly tied.
- **Ridge vs. lasso.** Ridge is slightly more accurate here, which fits
  predictors that are correlated groups of real signal rather than a few true
  ones plus noise. Lasso gives up a little accuracy for a model about half the
  size.

Results are deterministic: seeds 0..N−1 are shared across methods, so every
method is evaluated on the same splits.
