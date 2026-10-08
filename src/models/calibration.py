import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Tuple, Any

class BayesianIsotonicCalibrator:
    """
    Fits an Isotonic Regression on validation holdouts and applies
    Bayesian smoothing with dummy weight to prevent zero/one probability collapse.
    """
    def __init__(self, dummy_weight: float = 0.02, base_prior: float = 0.448):
        self.dummy_weight = dummy_weight
        self.base_prior = base_prior
        self.isotonic = IsotonicRegression(out_of_bounds='clip')

    def fit(self, raw_probs: np.ndarray, y_val: np.ndarray) -> "BayesianIsotonicCalibrator":
        self.isotonic.fit(raw_probs, y_val)
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        calibrated = self.isotonic.predict(raw_probs)
        # Bayesian shrinkage: (p_cal + w * prior) / (1 + w)
        smoothed = (calibrated + self.dummy_weight * self.base_prior) / (1.0 + self.dummy_weight)
        return smoothed

def calibrate_model(base_model: Any, X_train: Any, y_train: Any, X_val: Any, y_val: Any,
                    dummy_weight: float = 0.02, prior: float = 0.448) -> Tuple[Any, BayesianIsotonicCalibrator]:
    """Fits base model on training fold and calibrates on validation fold."""
    base_model.fit(X_train, y_train)
    raw_val = base_model.predict_proba(X_val)[:, 1]
    calibrator = BayesianIsotonicCalibrator(dummy_weight=dummy_weight, base_prior=prior)
    calibrator.fit(raw_val, y_val)
    return base_model, calibrator
