"""Compare every trained method.

Usage:
    python compare.py

Reads results/*/metrics.json and split_errors.csv, prints a sorted table, and
saves:
    results/comparison.csv           summary table
    results/comparison.png           test MSE distribution per method
    results/coefficients_compare.png OLS vs ridge vs lasso coefficients side by side
Because all methods share split seeds, "beats OLS" is a paired comparison.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["ols", "ridge", "lasso"]


def main():
    runs = {}
    for p in glob.glob("results/*/metrics.json"):
        with open(p) as f:
            m = json.load(f)
        d = os.path.dirname(p)
        runs[m["method"]] = (m, pd.read_csv(os.path.join(d, "split_errors.csv")),
                             pd.read_csv(os.path.join(d, "coefficients.csv"), index_col=0)["coef"])
    if not runs:
        raise SystemExit("No results found. Run train.py first.")
    names = [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER))

    base = runs["ols"][1].set_index("seed").test_mse if "ols" in runs else None
    rows = []
    for n in names:
        m, errs, _ = runs[n]
        row = {"method": n, "test_mse_mean": m["test_mse_mean"], "test_mse_sd": m["test_mse_sd"],
               "nonzero_coefs": m["n_nonzero_mean"], "lambda_mean": m["lambda_mean"]}
        if base is not None and n != "ols":
            e = errs.set_index("seed").test_mse.reindex(base.index)
            row["vs_ols_pct"] = 100 * (m["test_mse_mean"] / base.mean() - 1)
            row["beats_ols_pct"] = 100 * np.mean(e < base)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("test_mse_mean")
    table.to_csv("results/comparison.csv", index=False)
    print(table.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([runs[n][1].test_mse for n in names], tick_labels=names)
    ax.set_ylabel("test MSE (log salary)")
    ax.set_title(f"Test MSE over {runs[names[0]][0]['n_splits']} random 70/30 splits")
    fig.tight_layout(); fig.savefig("results/comparison.png", dpi=130); plt.close(fig)

    coefs = pd.DataFrame({n: runs[n][2] for n in names})
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.8 / len(names); x = np.arange(len(coefs))
    for i, n in enumerate(names):
        ax.bar(x + i * w - 0.4 + w / 2, coefs[n], w, label=n)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(coefs.index, rotation=90)
    ax.set_ylabel("standardized coefficient"); ax.legend()
    ax.set_title("Coefficients fit on all data: OLS vs ridge vs lasso")
    fig.tight_layout(); fig.savefig("results/coefficients_compare.png", dpi=130); plt.close(fig)
    print("saved results/comparison.csv, comparison.png, coefficients_compare.png")


if __name__ == "__main__":
    main()
