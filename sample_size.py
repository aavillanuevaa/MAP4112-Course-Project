"""Hypothesis test: does the benefit of regularization grow as n shrinks?

Usage:
    python sample_size.py
    python sample_size.py --reps 100 --sizes 30 40 60 100 150 200

For each training size n, draws `reps` random training sets of size n (the
rest of the 263 players are the test set), fits every method in methods/, and
records test MSE. Saves to results/sample_size/:
    sample_size.csv     every (n, seed, method, test_mse)
    summary.csv         median and IQR of test MSE per (n, method)
    sample_size.png     median test MSE vs training n, IQR shaded
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

import methods
from data import load_hitters, split

OUT = os.path.join("results", "sample_size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--sizes", type=int, nargs="+", default=[40, 60, 100, 150, 200])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    X, y = load_hitters()
    rows = []
    for n in args.sizes:
        for seed in range(args.reps):
            Xtr, Xte, ytr, yte = split(X, y, seed, train_size=n)
            for name in methods.METHODS:
                m = methods.get_method(name).fit(Xtr, ytr)
                rows.append({"n_train": n, "seed": seed, "method": name,
                             "test_mse": mean_squared_error(yte, m.predict(Xte))})
        print(f"n_train={n} done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sample_size.csv"), index=False)

    g = df.groupby(["n_train", "method"]).test_mse
    summary = pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)}).reset_index()
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(summary.pivot(index="n_train", columns="method", values="median").round(3))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in methods.METHODS:
        s = summary[summary.method == name]
        ax.plot(s.n_train, s["median"], "o-", label=name)
        ax.fill_between(s.n_train, s.q25, s.q75, alpha=0.15)
    ax.set_xlabel("training set size n (p = 19 predictors)")
    ax.set_ylabel("test MSE (median, IQR shaded)")
    ax.set_title("Regularization helps most when n is small")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sample_size.png"), dpi=130); plt.close(fig)
    print(f"saved to {OUT}/")


if __name__ == "__main__":
    main()
