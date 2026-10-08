import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.features.telephony import extract_telephony_features
from src.features.account import (
    extract_account_features,
    extract_field_visit_features,
    extract_payment_features
)
from src.utils.logger import get_logger

logger = get_logger("FeaturePipeline")

EXCLUDE_FROM_FEATURES: List[str] = [
    'phone_id', 'account_id', 'target_rpc', 'split',
    'has_rpc_disposition', 'verified_rpc', 'is_third_party', 'is_invalid',
    'added_date'
]

class FeaturePipeline:
    def __init__(self):
        self.feature_columns: List[str] = []

    def build_feature_matrix(self, raw_datasets: Dict[str, pd.DataFrame], labeled_phones: pd.DataFrame) -> pd.DataFrame:
        """Joins all domain features into a single feature matrix."""
        logger.info("Constructing complete feature matrix...")
        telephony_feats = extract_telephony_features(raw_datasets['dial_attempts'])
        account_feats = extract_account_features(raw_datasets['accounts'])
        field_feats = extract_field_visit_features(raw_datasets['field_visits'])
        payment_feats = extract_payment_features(raw_datasets['payments'])

        base_df = labeled_phones[[
            'phone_id', 'account_id', 'source', 'priority_slot',
            'target_rpc', 'is_third_party', 'is_invalid'
        ]].copy()
        base_df['source'] = base_df['source'].astype('category').cat.codes

        # Sequential left joins
        df_model = base_df.merge(telephony_feats, on='phone_id', how='left')
        df_model = df_model.merge(account_feats, on='account_id', how='left')
        df_model = df_model.merge(field_feats, on='account_id', how='left')
        df_model = df_model.merge(payment_feats, on='account_id', how='left')

        # Relational density feature
        df_model['phones_per_acct'] = df_model.groupby('account_id')['phone_id'].transform('count')

        # Clean null values in numeric feature space
        num_cols = df_model.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c not in ['target_rpc']]
        df_model[num_cols] = df_model[num_cols].fillna(-1.0)

        self.feature_columns = [c for c in df_model.columns if c not in EXCLUDE_FROM_FEATURES]
        logger.info(f"Feature matrix assembled: Shape={df_model.shape} | Predictors={len(self.feature_columns)}")
        return df_model

    def get_splits(self, df_model: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partitions feature matrix into train, validation, and test subsets based on account splits."""
        train_df = df_model[df_model['split'] == 'train'].copy()
        val_df = df_model[df_model['split'] == 'validation'].copy()
        test_df = df_model[df_model['split'] == 'test'].copy()

        logger.info(f"Split distributions -> Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
        return train_df, val_df, test_df
