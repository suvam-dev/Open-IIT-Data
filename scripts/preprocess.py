import os
import argparse
import pandas as pd

from src.config import load_config
from src.data.loader import DataLoader
from src.features.pipeline import FeaturePipeline
from src.utils.logger import get_logger

logger = get_logger("Preprocess")

def preprocess_data(config, output_dir="processed_data"):
    """
    Reads raw data, processes it into a feature matrix, 
    and saves the train/val/test splits to the output directory.
    """
    logger.info("=== PREPROCESSING STEP 1: INGESTING RAW DATA ===")
    loader = DataLoader(config)
    datasets = loader.load_raw_datasets()
    
    labeled_phones = loader.construct_target_labels(
        datasets['phones'], datasets['dial_attempts'], datasets['verified_contact_points']
    )

    logger.info("=== PREPROCESSING STEP 2: BUILDING FEATURE STORE ===")
    feature_pipe = FeaturePipeline()
    df_model = feature_pipe.build_feature_matrix(datasets, labeled_phones)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"=== PREPROCESSING STEP 3: SAVING PROCESSED DATA to {output_dir} ===")
    # Save the full processed dataset
    full_path = os.path.join(output_dir, "processed_features.csv")
    df_model.to_csv(full_path, index=False)
    logger.info(f"Saved full processed feature matrix to {full_path}")
    
    train_df, val_df, test_df = feature_pipe.get_splits(df_model)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    logger.info("Data preprocessing completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw data into feature matrices.")
    parser.add_argument("--output-dir", default="processed_data", help="Directory to save processed datasets.")
    args = parser.parse_args()
    
    config = load_config()
    preprocess_data(config, args.output_dir)
