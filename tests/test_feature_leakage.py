import pytest
from src.config import load_config
from src.data.loader import DataLoader
from src.features.pipeline import FeaturePipeline, EXCLUDE_FROM_FEATURES

def test_zero_target_leakage():
    """Asserts that no label derivatives or ground truth flags are present in feature space."""
    config = load_config()
    loader = DataLoader(config)
    datasets = loader.load_raw_datasets()
    labeled = loader.construct_target_labels(
        datasets['phones'], datasets['dial_attempts'], datasets['verified_contact_points']
    )
    
    pipe = FeaturePipeline()
    df_model = pipe.build_feature_matrix(datasets, labeled)
    
    leaky_patterns = ['rpc', 'verified', 'target', 'has_ptp']
    for feat in pipe.feature_columns:
        for leak in leaky_patterns:
            assert leak not in feat.lower(), f"Potential target leakage detected in feature: '{feat}'"

def test_split_account_isolation():
    """Asserts zero account overlap across train, validation, and test splits."""
    config = load_config()
    loader = DataLoader(config)
    datasets = loader.load_raw_datasets()
    splits_df = datasets['splits']
    
    train_accs = set(splits_df[splits_df['split'] == 'train']['account_id'])
    val_accs = set(splits_df[splits_df['split'] == 'validation']['account_id'])
    test_accs = set(splits_df[splits_df['split'] == 'test']['account_id'])
    
    assert len(train_accs & val_accs) == 0, "Account overlap between train and validation splits"
    assert len(train_accs & test_accs) == 0, "Account overlap between train and test splits"
    assert len(val_accs & test_accs) == 0, "Account overlap between validation and test splits"
