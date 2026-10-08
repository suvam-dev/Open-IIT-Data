import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve,
    precision_recall_curve, classification_report
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Any
from src.utils.logger import get_logger

logger = get_logger("ModelEvaluation")

def evaluate_models(models: List[Tuple[str, Any, Any]], X_test: pd.DataFrame, y_test: pd.Series,
                    reports_dir: str = "reports") -> pd.DataFrame:
    """Evaluates candidate models on unseen test split, generates comparison table & visual curves."""
    logger.info("Evaluating models on test dataset...")
    os.makedirs(reports_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_roc, ax_cal, ax_pr = axes
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    results = []

    for name, base_model, calibrator in models:
        raw_probs = base_model.predict_proba(X_test)[:, 1]
        cal_probs = calibrator.predict(raw_probs)

        auc_raw = roc_auc_score(y_test, raw_probs)
        auc_cal = roc_auc_score(y_test, cal_probs)
        brier_raw = brier_score_loss(y_test, raw_probs)
        brier_cal = brier_score_loss(y_test, cal_probs)

        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, cal_probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc_cal:.3f})")

        # Plot Calibration
        prob_true, prob_pred = calibration_curve(y_test, cal_probs, n_bins=10)
        ax_cal.plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier={brier_cal:.4f})")

        # Plot PR
        prec, rec, _ = precision_recall_curve(y_test, cal_probs)
        ax_pr.plot(rec, prec, label=f"{name}")

        results.append({
            'Model': name,
            'AUC (raw)': round(auc_raw, 4),
            'AUC (calibrated)': round(auc_cal, 4),
            'Brier (raw)': round(brier_raw, 4),
            'Brier (calibrated)': round(brier_cal, 4),
        })

    ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR'); ax_roc.set_title('ROC Curve'); ax_roc.legend(fontsize=8)
    ax_cal.set_xlabel('Mean Predicted Probability'); ax_cal.set_ylabel('Fraction of Positives')
    ax_cal.set_title('Calibration Curves'); ax_cal.legend(fontsize=8)
    ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision'); ax_pr.set_title('Precision-Recall Curve'); ax_pr.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "model_evaluation.png")
    plt.savefig(plot_path, dpi=120)
    # Also save to root for root backward-compatibility
    plt.savefig("model_evaluation.png", dpi=120)
    plt.close()
    logger.info(f"Saved evaluation plots to: {plot_path}")

    res_df = pd.DataFrame(results)
    return res_df
