"""Data checks for the report's data/preprocessing section.

Usage:
    python explore.py

Prints missing values, target skew, the most correlated predictor pairs and
VIFs, and saves to results/eda/:
    data_summary.txt    the printed checks
    target.png          Salary vs log(Salary) histograms
    correlation.png     predictor correlation heatmap
    vif.png             variance inflation factors (log scale)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ISLP import load_data

from data import load_hitters

OUT = os.path.join("results", "eda")


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = load_data("Hitters")
    X, y = load_hitters()
    salary = np.exp(y)

    corr = X.corr()
    c = corr.abs().values.copy(); np.fill_diagonal(c, 0)
    pairs = (pd.DataFrame(c, index=X.columns, columns=X.columns).stack()
             .sort_values(ascending=False).iloc[::2].head(8))
    vif = pd.Series(np.diag(np.linalg.inv(corr.values)), index=X.columns)

    lines = [
        f"Raw shape: {raw.shape}",
        f"Missing values: {raw.isna().sum()[raw.isna().sum() > 0].to_dict()}",
        f"After dropping missing Salary: {X.shape[0]} rows, {X.shape[1]} predictors",
        f"Salary ($1000s): min {salary.min():.1f}, median {salary.median():.1f}, max {salary.max():.1f}",
        f"Skewness: Salary {salary.skew():.2f}, log(Salary) {y.skew():.2f}",
        f"Condition number of standardized X: {np.linalg.cond((X - X.mean()) / X.std()):.0f}",
        "", "Most correlated predictor pairs (|r|):", pairs.round(3).to_string(),
        "", "VIF (>10 = severe collinearity):", vif.round(1).sort_values(ascending=False).to_string(),
    ]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "data_summary.txt"), "w") as f:
        f.write(text + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(salary, bins=30, color="#888"); ax[0].set_title(f"Salary ($1000s), skew {salary.skew():.2f}")
    ax[1].hist(y, bins=30, color="#4a7"); ax[1].set_title(f"log(Salary), skew {y.skew():.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "target.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Predictor correlations"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "correlation.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(vif.index, vif.values, color=["#c64" if v > 10 else "#888" for v in vif])
    ax.axhline(10, ls="--", c="k"); ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=90); ax.set_ylabel("VIF (log scale)")
    ax.set_title("Variance inflation factors (red: VIF > 10)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "vif.png"), dpi=130); plt.close(fig)
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()
