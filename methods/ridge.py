"""Ridge regression: minimize ||y - Xb||^2 + lambda * ||b||_2^2.

lambda (sklearn's `alpha`) is chosen by 10-fold cross-validation on the
training split over a log-spaced grid.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 5, 100)


def build_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=10))
