import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.calibration import calibrate_model, BayesianIsotonicCalibrator
from src.config import AppConfig, load_config, load_model_params
from src.utils.logger import get_logger

logger = get_logger("ModelTrainer")

class ModelTrainer:
    def __init__(self, config: AppConfig = None):
        self.config = config or load_config()
        self.model_params = load_model_params()

    def run_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Performs Stratified K-Fold Cross-Validation on candidate algorithms."""
        logger.info(f"Running {self.config.cv_n_splits}-fold Stratified Cross-Validation on training fold...")
        cv = StratifiedKFold(n_splits=self.config.cv_n_splits, shuffle=True, random_state=self.config.cv_random_state)
        
        cv_scores = {}
        models_to_test = {
            "Logistic Regression": LogisticRegression(**self.model_params['baseline_models']['logistic_regression']),
            "Random Forest": RandomForestClassifier(**self.model_params['candidate_models']['random_forest']),
            "HistGradientBoosting": HistGradientBoostingClassifier(**self.model_params['candidate_models']['hist_gradient_boosting'])
        }

        for name, model in models_to_test.items():
            scores = cross_val_score(model, X_train.fillna(-1.0), y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_scores[name] = float(scores.mean())
            logger.info(f"CV ROC-AUC: {name:<22} = {scores.mean():.4f} (+/- {scores.std():.4f})")
            
        return cv_scores

    def train_and_calibrate_candidates(self, X_train: pd.DataFrame, y_train: pd.Series,
                                       X_val: pd.DataFrame, y_val: pd.Series) -> List[Tuple[str, Any, BayesianIsotonicCalibrator]]:
        """Trains all candidate architectures on train set and isotonically calibrates them on validation fold."""
        logger.info("Training and calibrating candidate models...")
        prior = self.config.base_rpc_prior
        w = self.config.dummy_weight

        # 1. Baseline: Logistic Regression
        lr_base = LogisticRegression(**self.model_params['baseline_models']['logistic_regression'])
        lr_base, lr_cal = calibrate_model(lr_base, X_train.fillna(-1.0), y_train, X_val.fillna(-1.0), y_val, dummy_weight=w, prior=prior)

        # 2. Candidate: Random Forest
        rf_base = RandomForestClassifier(**self.model_params['candidate_models']['random_forest'])
        rf_base, rf_cal = calibrate_model(rf_base, X_train.fillna(-1.0), y_train, X_val.fillna(-1.0), y_val, dummy_weight=w, prior=prior)

        # 3. Champion Candidate: HistGradientBoosting
        hgb_base = HistGradientBoostingClassifier(**self.model_params['candidate_models']['hist_gradient_boosting'])
        hgb_base, hgb_cal = calibrate_model(hgb_base, X_train, y_train, X_val.fillna(-1.0), y_val, dummy_weight=w, prior=prior)

        trained_candidates = [
            ("Logistic Regression (calibrated)", lr_base, lr_cal),
            ("Random Forest (calibrated)", rf_base, rf_cal),
            ("HistGradBoost (calibrated)", hgb_base, hgb_cal),
        ]
        return trained_candidates
