import os
import joblib
from typing import Dict, Any, List, Optional
from src.utils.logger import get_logger

logger = get_logger("ModelRegistry")

class ModelRegistry:
    def __init__(self, artifacts_dir: str = "artifacts/models"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def save_model(self, base_model: Any, calibrator: Any, feature_columns: List[str],
                   model_name: str = "champion_model", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Serializes trained model bundle into a joblib artifact."""
        artifact = {
            "base_model": base_model,
            "calibrator": calibrator,
            "feature_columns": feature_columns,
            "metadata": metadata or {}
        }
        filepath = os.path.join(self.artifacts_dir, f"{model_name}.joblib")
        joblib.dump(artifact, filepath)
        logger.info(f"Model artifact persisted successfully to: {filepath}")
        return filepath

    def load_model(self, model_name: str = "champion_model") -> Dict[str, Any]:
        """Loads serialized model bundle from artifacts directory."""
        filepath = os.path.join(self.artifacts_dir, f"{model_name}.joblib")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No persisted model found at: {filepath}")
        artifact = joblib.load(filepath)
        logger.info(f"Loaded model artifact from: {filepath}")
        return artifact
