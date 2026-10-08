import os
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import AppConfig, load_config
from src.data.schema import validate_dataframe, RPC_DISPOSITIONS, FAILED_DISPOSITIONS
from src.utils.logger import get_logger

logger = get_logger("DataLoader")

class DataLoader:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()

    def resolve_path(self, filename: str, is_shared: bool = False) -> str:
        """Finds file path checking configured directories and standard relative locations."""
        primary_dir = self.config.shared_data_dir if is_shared else self.config.ps2_data_dir
        path = os.path.join(primary_dir, filename)
        if os.path.exists(path):
            return path
        
        # Fallback candidate search
        candidates = [
            os.path.join("raw_data", filename),
            os.path.join("data", filename),
            filename,
            os.path.join("shared", filename),
            os.path.join("..", "shared", filename) if is_shared else filename,
            os.path.join("ps2_right_party_contact", filename)
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"Could not locate '{filename}' in primary '{primary_dir}' or fallback paths.")

    def load_raw_datasets(self) -> Dict[str, pd.DataFrame]:
        """Loads and validates all 8 core tabular datasets."""
        logger.info("Loading raw collections datasets...")
        datasets = {
            'accounts': pd.read_csv(self.resolve_path('accounts.csv', is_shared=True)),
            'phones': pd.read_csv(self.resolve_path('phones.csv', is_shared=False)),
            'dial_attempts': pd.read_csv(self.resolve_path('dial_attempts.csv', is_shared=True)),
            'field_visits': pd.read_csv(self.resolve_path('field_visits.csv', is_shared=True)),
            'payments': pd.read_csv(self.resolve_path('payments.csv', is_shared=True)),
            'skip_traces': pd.read_csv(self.resolve_path('skip_traces.csv', is_shared=False)),
            'verified_contact_points': pd.read_csv(self.resolve_path('verified_contact_points.csv', is_shared=False)),
            'splits': pd.read_csv(self.resolve_path('splits.csv', is_shared=True)),
        }

        for name, df in datasets.items():
            validate_dataframe(df, name)
            logger.info(f"Loaded {name:<24} | Rows: {len(df):>6} | Cols: {len(df.columns):>2}")

        # Attach deterministic split mapping to accounts
        datasets['accounts'] = datasets['accounts'].merge(datasets['splits'], on='account_id', how='left')
        return datasets

    def load_merged_raw_data(self, filename: str = "merged_raw_data.csv") -> pd.DataFrame:
        """Loads pre-merged consolidated raw dataset containing all phone, account, telephony, and outcome features."""
        filepath = self.resolve_path(filename, is_shared=False)
        logger.info(f"Loading single consolidated raw dataset from '{filepath}'...")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded merged raw dataset: Shape={df.shape} | Rows={len(df)} | Columns={len(df.columns)}")
        return df

    def construct_target_labels(self, phones: pd.DataFrame, dial_attempts: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs leakage-free binary target label: target_rpc in {0, 1}.
        target_rpc = 1 if any dial attempt recorded an RPC disposition OR verified as borrower number.
        """
        logger.info("Constructing ground-truth RPC target labels...")
        
        # Phone level dial attempt RPC history (Used ONLY for label creation)
        phone_rpc_agg = dial_attempts.groupby('phone_id').agg(
            has_rpc_disposition=('disposition', lambda x: x.isin(RPC_DISPOSITIONS).any().astype(int))
        ).reset_index()

        # Audit verifications
        verified_clean = verified.copy()
        verified_clean['verified_rpc'] = (verified_clean['verified_status'] == 'borrower_number').astype(int)
        verified_clean['is_third_party'] = (verified_clean['verified_status'] == 'third_party_number').astype(int)
        verified_clean['is_invalid'] = verified_clean['verified_status'].isin(
            ['not_borrower_number', 'switched_off', 'invalid_number']
        ).astype(int)

        labeled_phones = phones[['phone_id', 'account_id', 'source', 'priority_slot', 'added_date']].merge(
            phone_rpc_agg, on='phone_id', how='left'
        ).merge(
            verified_clean[['phone_id', 'verified_rpc', 'is_third_party', 'is_invalid']],
            on='phone_id', how='left'
        )

        for col in ['has_rpc_disposition', 'verified_rpc', 'is_third_party', 'is_invalid']:
            labeled_phones[col] = labeled_phones[col].fillna(0).astype(int)

        # Final target variable
        labeled_phones['target_rpc'] = np.maximum(
            labeled_phones['has_rpc_disposition'],
            labeled_phones['verified_rpc']
        ).astype(int)

        rpc_rate = labeled_phones['target_rpc'].mean()
        logger.info(f"Target label generated: Total={len(labeled_phones)} | RPC Rate={rpc_rate:.4f} ({labeled_phones['target_rpc'].sum()} positives)")
        return labeled_phones
