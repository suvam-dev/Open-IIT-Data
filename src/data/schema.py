from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd

RPC_DISPOSITIONS: List[str] = [
    'rpc_call_back', 'rpc_hardship', 'rpc_refused',
    'rpc_ptp', 'rpc_hung_up', 'rpc_dispute', 'rpc_claims_paid'
]

FAILED_DISPOSITIONS: List[str] = [
    'switched_off', 'not_reachable', 'invalid_number', 'wrong_number'
]

REQUIRED_COLUMNS: Dict[str, List[str]] = {
    'accounts': ['account_id', 'outstanding', 'dpd_start', 'emi_amount', 'bureau_score_band'],
    'phones': ['phone_id', 'account_id', 'source', 'priority_slot'],
    'dial_attempts': ['attempt_id', 'phone_id', 'account_id', 'disposition', 'ring_duration_s', 'talk_duration_s'],
    'field_visits': ['visit_id', 'account_id', 'outcome', 'dwell_s'],
    'payments': ['payment_id', 'account_id', 'amount'],
    'skip_traces': ['trace_id', 'account_id', 'result', 'cost_inr'],
    'verified_contact_points': ['phone_id', 'account_id', 'verified_status'],
    'splits': ['account_id', 'split'],
}

def validate_dataframe(df: pd.DataFrame, table_name: str) -> None:
    """Asserts that all critical columns exist in the DataFrame."""
    if table_name not in REQUIRED_COLUMNS:
        return
    required = REQUIRED_COLUMNS[table_name]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Table '{table_name}' is missing required columns: {missing}")
