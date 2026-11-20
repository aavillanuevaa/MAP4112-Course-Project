"""Ordinary least squares: minimize ||y - Xb||^2. No tuning parameter."""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = None  # no penalty, so no coefficient path


def build_model():
    return make_pipeline(StandardScaler(), LinearRegression())
