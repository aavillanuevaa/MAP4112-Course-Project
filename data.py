"""Shared data loading and splitting for the Hitters dataset.

Hitters (James et al., *An Introduction to Statistical Learning*) has 1986 MLB
salaries and 1986/career batting stats for 322 players. We:
  - drop the 59 players with missing Salary  -> 263 rows
  - one-hot encode League, Division, NewLeague (drop_first)  -> 19 predictors
  - model log(Salary), since raw Salary is strongly right-skewed

Standardization is NOT done here. Each method's pipeline (see methods/) fits a
StandardScaler on the training split only, so no test information leaks in.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ISLP import load_data

TEST_SIZE = 0.3


def load_hitters():
    """Return (X, y): X is a float DataFrame of 19 predictors, y is log(Salary)."""
    df = load_data("Hitters").dropna()
    y = np.log(df["Salary"]).rename("logSalary")
    X = pd.get_dummies(df.drop(columns="Salary"), drop_first=True).astype(float)
    return X, y


def split(X, y, seed, train_size=None):
    """Random train/test split. Pass train_size (int) to fix the training n."""
    if train_size is not None:
        return train_test_split(X, y, train_size=train_size, random_state=seed)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
