"""Regenerate one method's plots from its saved CSVs, without refitting.

Usage:
    python evaluate.py --method lasso

Reads results/<method>/*.csv and writes:
    coefficients.png    full-data standardized coefficients
    split_errors.png    test MSE across random splits
    coef_path.png       coefficient path vs lambda (ridge/lasso only)
    cv_curve.png        10-fold CV error vs lambda (ridge/lasso only)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TITLES = {"ols": "Least squares", "ridge": "Ridge", "lasso": "Lasso"}


def make_plots(method):
    d = os.path.join("results", method)
    name = TITLES.get(method, method)
    with open(os.path.join(d, "metrics.json")) as f:
        m = json.load(f)

    coefs = pd.read_csv(os.path.join(d, "coefficients.csv"), index_col=0)["coef"]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#bbb" if abs(v) <= 1e-10 else ("#c64" if v < 0 else "#36a") for v in coefs]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, c="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("standardized coefficient")
    ax.set_title(f"{name}: coefficients (fit on all data)")
    fig.tight_layout(); fig.savefig(os.path.join(d, "coefficients.png"), dpi=130); plt.close(fig)

    errs = pd.read_csv(os.path.join(d, "split_errors.csv"))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(errs.seed, errs.test_mse, "o-", ms=3)
    ax.axhline(m["test_mse_mean"], ls="--", c="k", label=f"mean {m['test_mse_mean']:.3f}")
    ax.set_xlabel("split (seed)"); ax.set_ylabel("test MSE"); ax.legend()
    ax.set_title(f"{name}: test MSE across {m['n_splits']} random 70/30 splits")
    fig.tight_layout(); fig.savefig(os.path.join(d, "split_errors.png"), dpi=130); plt.close(fig)

    if os.path.exists(os.path.join(d, "path.csv")):
        path = pd.read_csv(os.path.join(d, "path.csv"))
        lam = path.pop("lambda")
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(path.columns):
            ax.plot(lam, path[c], label=c, color=cmap(i % 20))
        ax.axvline(m["full_data_lambda"], ls="--", c="k", label="CV-chosen λ")
        ax.set_xscale("log"); ax.set_xlabel("λ"); ax.set_ylabel("standardized coefficient")
        ax.set_title(f"{name}: coefficient paths")
        ax.legend(fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.tight_layout(); fig.savefig(os.path.join(d, "coef_path.png"), dpi=130); plt.close(fig)

        cv = pd.read_csv(os.path.join(d, "cv_curve.csv"))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(cv["lambda"], cv.cv_mse, yerr=cv.cv_se, fmt="-", c="#36a", ecolor="#bbb")
        ax.axvline(m["full_data_lambda"], ls="--", c="k", label="CV-chosen λ")
        ax.set_xscale("log"); ax.set_xlabel("λ"); ax.set_ylabel("10-fold CV MSE"); ax.legend()
        ax.set_title(f"{name}: cross-validation error vs λ")
        fig.tight_layout(); fig.savefig(os.path.join(d, "cv_curve.png"), dpi=130); plt.close(fig)

    print(f"[{method}] plots saved to {d}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    make_plots(ap.parse_args().method)
