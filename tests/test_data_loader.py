import pytest
import pandas as pd
from src.config import load_config
from src.data.loader import DataLoader
from src.data.schema import validate_dataframe

def test_data_loader_integrity():
    config = load_config()
    loader = DataLoader(config)
    datasets = loader.load_raw_datasets()
    
    assert 'accounts' in datasets
    assert 'phones' in datasets
    assert len(datasets['accounts']) == 2400
    assert len(datasets['phones']) == 5719

def test_target_label_generation():
    config = load_config()
    loader = DataLoader(config)
    datasets = loader.load_raw_datasets()
    labeled = loader.construct_target_labels(
        datasets['phones'], datasets['dial_attempts'], datasets['verified_contact_points']
    )
    
    assert 'target_rpc' in labeled.columns
    assert set(labeled['target_rpc'].unique()).issubset({0, 1})
    assert len(labeled) == 5719

def test_schema_validation_failure():
    invalid_df = pd.DataFrame({"random_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe(invalid_df, "accounts")

def test_load_merged_raw_data():
    config = load_config()
    loader = DataLoader(config)
    merged_df = loader.load_merged_raw_data()
    assert len(merged_df) == 5719
    assert 'phone_id' in merged_df.columns
    assert 'account_id' in merged_df.columns
    assert 'target_rpc' in merged_df.columns
    assert 'split' in merged_df.columns
    assert merged_df.shape[1] >= 50

