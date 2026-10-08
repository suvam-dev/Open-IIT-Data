import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import yaml

@dataclass
class AppConfig:
    raw: Dict[str, Any]
    
    # Paths
    raw_data_dir: str
    ps2_data_dir: str
    shared_data_dir: str
    artifacts_dir: str
    outputs_dir: str
    reports_dir: str
    
    # Business Constants
    p_find_borrower_trace: float
    p_meet_borrower_visit: float
    avg_trace_cost_inr: float
    p_recovery_given_rpc: float
    collection_fraction: float
    call_cost_inr: float
    field_visit_cost_inr: float
    
    # Calibration & Thresholds
    calibration_method: str
    dummy_weight: float
    base_rpc_prior: float
    t_high: float
    t_opt: float
    t_low: float
    
    # CV
    cv_n_splits: int
    cv_random_state: int

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads configuration YAML file and returns typed AppConfig dataclass."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "configs", "config.yaml")
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    paths = cfg.get("paths", {})
    b_const = cfg.get("business_constants", {})
    calib = cfg.get("calibration", {})
    thresh = cfg.get("health_thresholds", {})
    cv_cfg = cfg.get("cv", {})
    
    return AppConfig(
        raw=cfg,
        raw_data_dir=paths.get("raw_data_dir", "raw_data"),
        ps2_data_dir=paths.get("ps2_data_dir", "raw_data"),
        shared_data_dir=paths.get("shared_data_dir", "raw_data"),
        artifacts_dir=paths.get("artifacts_dir", "artifacts/models"),
        outputs_dir=paths.get("outputs_dir", "outputs"),
        reports_dir=paths.get("reports_dir", "reports"),
        p_find_borrower_trace=b_const.get("p_find_borrower_trace", 0.228),
        p_meet_borrower_visit=b_const.get("p_meet_borrower_visit", 0.216),
        avg_trace_cost_inr=b_const.get("avg_trace_cost_inr", 104.0),
        p_recovery_given_rpc=b_const.get("p_recovery_given_rpc", 0.05),
        collection_fraction=b_const.get("collection_fraction", 0.10),
        call_cost_inr=b_const.get("call_cost_inr", 5.0),
        field_visit_cost_inr=b_const.get("field_visit_cost_inr", 200.0),
        calibration_method=calib.get("method", "isotonic"),
        dummy_weight=calib.get("dummy_weight", 0.02),
        base_rpc_prior=calib.get("base_rpc_prior", 0.448),
        t_high=thresh.get("t_high", 0.70),
        t_opt=thresh.get("t_opt", 0.40),
        t_low=thresh.get("t_low", 0.15),
        cv_n_splits=cv_cfg.get("n_splits", 5),
        cv_random_state=cv_cfg.get("random_state", 42),
    )

def load_model_params(params_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads candidate model parameters and hyperparameter search grids."""
    if params_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        params_path = os.path.join(base_dir, "configs", "model_params.yaml")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)
