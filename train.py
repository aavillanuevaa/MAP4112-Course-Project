"""Fit one method over repeated random train/test splits and save its results.

Usage:
    python train.py --method ridge
    python train.py --method lasso --reps 100

Writes to results/<method>/:
    metrics.json        mean/sd test MSE, mean chosen lambda, mean # nonzero coefs
    split_errors.csv    test MSE, chosen lambda and # nonzero coefs for every split
    coefficients.csv    standardized coefficients from a fit on all 263 players
    path.csv            coefficient path over lambda (ridge/lasso only)
    cv_curve.csv        10-fold CV error vs lambda on the full data (ridge/lasso only)
and then calls evaluate.py to draw the plots.

Every method uses the same seeds 0..reps-1, so split i is identical across
methods and the comparison in compare.py is paired.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, lasso_path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import methods
from data import load_hitters, split
from evaluate import make_plots


def chosen_alpha(model):
    est = model[-1]
    return float(getattr(est, "alpha_", 0.0))


def cv_curve(name, Xs, y, alphas):
    """10-fold CV mean/sd MSE for each lambda (data already standardized)."""
    kf = KFold(10, shuffle=True, random_state=0)
    Est = Ridge if name == "ridge" else Lasso
    rows = []
    for a in alphas:
        errs = []
        for tr, te in kf.split(Xs):
            m = Est(alpha=a, max_iter=50000).fit(Xs[tr], y[tr])
            errs.append(mean_squared_error(y[te], m.predict(Xs[te])))
        rows.append({"lambda": a, "cv_mse": np.mean(errs), "cv_se": np.std(errs) / np.sqrt(len(errs))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(methods.METHODS))
    ap.add_argument("--reps", type=int, default=50, help="number of random splits")
    args = ap.parse_args()

    X, y = load_hitters()
    out = os.path.join("results", args.method)
    os.makedirs(out, exist_ok=True)

    # 1) Repeated train/test splits
    rows = []
    for seed in range(args.reps):
        Xtr, Xte, ytr, yte = split(X, y, seed)
        model = methods.get_method(args.method).fit(Xtr, ytr)
        rows.append({
            "seed": seed,
            "test_mse": mean_squared_error(yte, model.predict(Xte)),
            "lambda": chosen_alpha(model),
            "n_nonzero": int(np.sum(np.abs(model[-1].coef_) > 1e-10)),
        })
    errs = pd.DataFrame(rows)
    errs.to_csv(os.path.join(out, "split_errors.csv"), index=False)

    # 2) Fit on all data for interpretable coefficients
    full = methods.get_method(args.method).fit(X, y)
    coefs = pd.Series(full[-1].coef_, index=X.columns, name="coef")
    coefs.to_csv(os.path.join(out, "coefficients.csv"))

    # 3) Coefficient path and CV curve (penalized methods only)
    if args.method in ("ridge", "lasso"):
        Xs = StandardScaler().fit_transform(X)
        yv = y.values
        if args.method == "ridge":
            alphas = methods.ridge.ALPHAS
            path = np.array([Ridge(alpha=a).fit(Xs, yv).coef_ for a in alphas])
        else:
            alphas = full[-1].alphas_
            alphas, path, _ = lasso_path(Xs, yv - yv.mean(), alphas=alphas, max_iter=50000)
            path = path.T
        pd.DataFrame(path, columns=X.columns).assign(**{"lambda": alphas}).to_csv(
            os.path.join(out, "path.csv"), index=False)
        cv_curve(args.method, Xs, yv, alphas).to_csv(os.path.join(out, "cv_curve.csv"), index=False)

    metrics = {
        "method": args.method,
        "n_splits": args.reps,
        "test_size": 0.3,
        "test_mse_mean": errs.test_mse.mean(),
        "test_mse_sd": errs.test_mse.std(),
        "lambda_mean": errs["lambda"].mean(),
        "n_nonzero_mean": errs.n_nonzero.mean(),
        "n_features": X.shape[1],
        "full_data_lambda": chosen_alpha(full),
        "full_data_zeroed": [c for c, v in coefs.items() if abs(v) <= 1e-10],
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{args.method}] test MSE {metrics['test_mse_mean']:.4f} ± {metrics['test_mse_sd']:.4f} "
          f"over {args.reps} splits | nonzero coefs {metrics['n_nonzero_mean']:.1f}/{X.shape[1]}")
    make_plots(args.method)


if __name__ == "__main__":
    main()
