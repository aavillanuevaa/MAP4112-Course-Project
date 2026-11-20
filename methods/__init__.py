"""Registry of regression methods. Add a new method by creating
methods/<name>.py with a build_model() function and registering it here."""
from . import ols, ridge, lasso

METHODS = {
    "ols": ols.build_model,
    "ridge": ridge.build_model,
    "lasso": lasso.build_model,
}


def get_method(name):
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Choose from: {list(METHODS)}")
    return METHODS[name]()
