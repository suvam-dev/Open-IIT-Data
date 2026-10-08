import argparse
import sys
import os
import pandas as pd

from src.config import load_config
from src.data.loader import DataLoader
from src.features.pipeline import FeaturePipeline
from src.models.train import ModelTrainer
from src.models.evaluate import evaluate_models
from src.models.registry import ModelRegistry
from src.decision.engine import DecisionEngine
from src.decision.trace import SkipTraceOptimizer
from src.utils.logger import get_logger

logger = get_logger("CreditNirvanaCLI")

def run_train(config):
    """Executes data loading, feature engineering, CV, candidate training, evaluation, and model persistence."""
    logger.info("=== STEP 1: INGESTING RAW DATA ===")
    loader = DataLoader(config)
    datasets = loader.load_raw_datasets()
    labeled_phones = loader.construct_target_labels(
        datasets['phones'], datasets['dial_attempts'], datasets['verified_contact_points']
    )

    logger.info("=== STEP 2: BUILDING FEATURE STORE ===")
    feature_pipe = FeaturePipeline()
    df_model = feature_pipe.build_feature_matrix(datasets, labeled_phones)
    train_df, val_df, test_df = feature_pipe.get_splits(df_model)

    features = feature_pipe.feature_columns
    X_train, y_train = train_df[features], train_df['target_rpc']
    X_val, y_val = val_df[features], val_df['target_rpc']
    X_test, y_test = test_df[features], test_df['target_rpc']

    logger.info("=== STEP 3: CROSS-VALIDATION & CANDIDATE TRAINING ===")
    trainer = ModelTrainer(config)
    trainer.run_cross_validation(X_train, y_train)
    candidates = trainer.train_and_calibrate_candidates(X_train, y_train, X_val, y_val)

    logger.info("=== STEP 4: MODEL EVALUATION ON UNSEEN TEST FOLD ===")
    results_df = evaluate_models(candidates, X_test.fillna(-1.0), y_test, reports_dir=config.reports_dir)
    print("\n" + results_df.to_string(index=False) + "\n")
    
    # Save benchmark table
    results_df.to_csv(os.path.join(config.outputs_dir, "model_comparison.csv"), index=False)
    results_df.to_csv("model_comparison.csv", index=False)

    logger.info("=== STEP 5: MODEL SELECTION & PERSISTENCE ===")
    # HistGradBoost is the designated champion
    champion_name, champion_base, champion_cal = candidates[2]
    registry = ModelRegistry(config.artifacts_dir)
    registry.save_model(
        base_model=champion_base,
        calibrator=champion_cal,
        feature_columns=features,
        model_name="champion_model",
        metadata={"champion": champion_name, "test_metrics": results_df.to_dict(orient="records")}
    )
    logger.info("Model training & persistence complete.")
    return df_model, champion_base, champion_cal, features

def run_predict(config, df_model=None, champion_base=None, champion_cal=None, features=None):
    """Loads champion model from registry and executes end-to-end decision engine."""
    logger.info("=== STEP 6: INFERENCE & DECISION ENGINE ===")
    registry = ModelRegistry(config.artifacts_dir)
    
    if champion_base is None or champion_cal is None or features is None:
        bundle = registry.load_model("champion_model")
        champion_base = bundle["base_model"]
        champion_cal = bundle["calibrator"]
        features = bundle["feature_columns"]

    if df_model is None:
        loader = DataLoader(config)
        datasets = loader.load_raw_datasets()
        labeled_phones = loader.construct_target_labels(
            datasets['phones'], datasets['dial_attempts'], datasets['verified_contact_points']
        )
        feature_pipe = FeaturePipeline()
        df_model = feature_pipe.build_feature_matrix(datasets, labeled_phones)

    # 1. Inference with Bayesian Isotonic Calibration
    raw_probs = champion_base.predict_proba(df_model[features].fillna(-1.0))[:, 1]
    df_model['rpc_probability'] = champion_cal.predict(raw_probs)

    # 2. Tier 1: Phone-level decision engine
    decision_engine = DecisionEngine(config)
    df_contacts = decision_engine.process_contact_recommendations(df_model)

    # 3. Tier 2: Account-level skip-trace engine
    trace_optimizer = SkipTraceOptimizer(config)
    df_accounts = trace_optimizer.optimize_accounts(df_contacts)

    # 4. Export formatted presentation deliverables
    os.makedirs(config.outputs_dir, exist_ok=True)

    contacts_out = df_contacts[[
        'account_id', 'phone_id', 'source', 'priority_slot',
        'rpc_probability', 'contact_health', 'cp_rank',
        'cp_action', 'ev_call', 'cp_reason',
        'total_attempts', 'wrong_number_count', 'switched_off_count', 'avg_talk_s',
        'outstanding', 'split'
    ]].rename(columns={
        'phone_id': 'contact_point_id',
        'source': 'phone_source',
        'cp_rank': 'priority_rank',
        'cp_action': 'recommended_action',
        'ev_call': 'expected_value_call_inr',
        'cp_reason': 'reason'
    })

    accounts_out = df_accounts[[
        'account_id', 'outstanding', 'n_phones', 'n_callable',
        'best_rpc_prob', 'best_callable_phone', 'ev_of_trace',
        'skip_trace_decision', 'skip_trace_reasoning', 'skip_trace_score'
    ]].rename(columns={
        'best_rpc_prob': 'best_available_rpc_probability',
        'best_callable_phone': 'recommended_first_contact',
        'ev_of_trace': 'expected_value_of_trace_inr'
    })

    contacts_out.to_csv(os.path.join(config.outputs_dir, "ps2_contact_recommendations.csv"), index=False)
    contacts_out.to_csv("ps2_contact_recommendations.csv", index=False)
    accounts_out.to_csv(os.path.join(config.outputs_dir, "ps2_account_decisions.csv"), index=False)
    accounts_out.to_csv("ps2_account_decisions.csv", index=False)

    logger.info(f"Generated {len(contacts_out)} contact-level recommendations and {len(accounts_out)} account-level decisions.")
    return contacts_out, accounts_out

def main():
    parser = argparse.ArgumentParser(description="CreditNirvana PS2 Production Machine Learning Pipeline")
    parser.add_argument("mode", choices=["pipeline", "train", "predict"], default="pipeline", nargs="?",
                        help="Execution mode: 'pipeline' (end-to-end), 'train' (training & evaluation), 'predict' (batch inference)")
    args = parser.parse_args()

    config = load_config()

    if args.mode == "train":
        run_train(config)
    elif args.mode == "predict":
        run_predict(config)
    else:
        df_model, champion_base, champion_cal, features = run_train(config)
        run_predict(config, df_model, champion_base, champion_cal, features)

if __name__ == "__main__":
    main()
