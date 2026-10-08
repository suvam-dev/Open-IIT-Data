import numpy as np
import pytest
from src.models.calibration import BayesianIsotonicCalibrator

def test_calibrator_bounds_and_smoothing():
    """Asserts that Bayesian dummy weight strictly prevents 0.0 and 1.0 probability collapse."""
    calibrator = BayesianIsotonicCalibrator(dummy_weight=0.02, base_prior=0.448)
    
    # Synthetic calibration fold: raw in [0, 1], y in {0, 1}
    raw_val = np.array([0.05, 0.1, 0.2, 0.5, 0.8, 0.95])
    y_val = np.array([0, 0, 0, 1, 1, 1])
    calibrator.fit(raw_val, y_val)
    
    # Test extreme raw values
    test_raw = np.array([0.0, 0.01, 0.5, 0.99, 1.0])
    probs = calibrator.predict(test_raw)
    
    # Monotonicity check
    assert np.all(np.diff(probs) >= 0), "Calibrated probabilities must be monotonically non-decreasing"
    
    # Strict non-zero and non-one checks
    assert np.all(probs > 0.0), "Calibrated probability collapsed to hard zero"
    assert np.all(probs < 1.0), "Calibrated probability collapsed to hard one"
    assert probs[0] >= 0.005, f"Lower bound should be safely smoothed, got {probs[0]}"
