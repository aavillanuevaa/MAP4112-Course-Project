"""Lasso: minimize (1/2n)||y - Xb||^2 + lambda * ||b||_1.

lambda (sklearn's `alpha`) is chosen by 10-fold cross-validation on the
training split over sklearn's automatic grid of 100 values.
Note sklearn scales the lasso loss by 1/(2n), so its lambda values are not on
the same scale as ridge's.
"""
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = None  # LassoCV builds its own grid


def build_model():
    return make_pipeline(StandardScaler(), LassoCV(cv=10, random_state=0, max_iter=50000))
