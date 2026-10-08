import pandas as pd
import numpy as np
from typing import List
from src.data.schema import FAILED_DISPOSITIONS
from src.utils.logger import get_logger

logger = get_logger("TelephonyFeatures")

def extract_telephony_features(dial_attempts: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts strictly leakage-safe operational call features per phone_id.
    EXCLUDES all target derivatives (has_rpc, rpc_count, rpc_rate, has_ptp).
    """
    logger.info("Extracting leakage-free telephony features...")
    df_dials = dial_attempts.copy()
    df_dials['attempt_ts'] = pd.to_datetime(df_dials['attempt_ts'])
    max_date = df_dials['attempt_ts'].max()
    df_dials['days_since'] = (max_date - df_dials['attempt_ts']).dt.total_seconds() / 86400.0

    feats = df_dials.groupby('phone_id').agg(
        total_attempts       = ('attempt_id', 'count'),
        wrong_number_count   = ('disposition', lambda x: x.isin(['wrong_number']).sum()),
        switched_off_count   = ('disposition', lambda x: x.isin(['switched_off']).sum()),
        not_reachable_count  = ('disposition', lambda x: x.isin(['not_reachable']).sum()),
        invalid_number_count = ('disposition', lambda x: x.isin(['invalid_number']).sum()),
        avg_ring_s           = ('ring_duration_s', 'mean'),
        avg_talk_s           = ('talk_duration_s', 'mean'),
        max_talk_s           = ('talk_duration_s', 'max'),
        pct_zero_talk        = ('talk_duration_s', lambda x: (x == 0).mean()),
        pct_zero_ring        = ('ring_duration_s', lambda x: (x == 0).mean()),
        n_agents             = ('agent_id', 'nunique'),
        n_channels           = ('channel', 'nunique'),
        days_since_last      = ('days_since', 'min'),
        days_since_first     = ('days_since', 'max'),
        attempts_7d          = ('days_since', lambda x: (x <= 7).sum()),
        attempts_30d         = ('days_since', lambda x: (x <= 30).sum()),
        recent_failures      = ('disposition', lambda x: x.head(5).isin(FAILED_DISPOSITIONS).sum()),
    ).reset_index()

    feats['failure_rate'] = (
        (feats['wrong_number_count']
         + feats['switched_off_count']
         + feats['not_reachable_count'])
        / feats['total_attempts'].clip(lower=1)
    )
    feats['talk_rate'] = (feats['avg_talk_s'] > 0).astype(int)

    logger.info(f"Telephony features extracted for {len(feats)} phone numbers.")
    return feats
