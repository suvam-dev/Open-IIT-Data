from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from src.utils.logger import get_logger

logger = get_logger("ModelTuner")

def tune_hist_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series,
                                param_grid: Dict[str, Any], n_splits: int = 3,
                                random_state: int = 42) -> Tuple[HistGradientBoostingClassifier, Dict[str, Any]]:
    """Runs cross-validated hyperparameter tuning for HistGradientBoostingClassifier."""
    logger.info("Executing cross-validated hyperparameter optimization...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    base_estimator = HistGradientBoostingClassifier(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Tuning complete. Best ROC-AUC: {grid_search.best_score_:.4f} with params: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_
