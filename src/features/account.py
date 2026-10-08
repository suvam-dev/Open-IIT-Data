import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("AccountFeatures")

def extract_account_features(accounts: pd.DataFrame) -> pd.DataFrame:
    """Extracts and encodes borrower-level financial and risk profile features."""
    logger.info("Extracting account-level features...")
    cols = [
        'account_id', 'outstanding', 'dpd_start', 'emi_amount',
        'bureau_score_band', 'other_active_loans', 'prev_ptp_count',
        'prev_ptp_broken', 'split', 'ability_to_pay_estimate',
        'lender_id', 'income_type', 'dialling_arm'
    ]
    feats = accounts[cols].copy()
    for cat_col in ['bureau_score_band', 'lender_id', 'income_type', 'dialling_arm']:
        feats[cat_col] = feats[cat_col].astype('category').cat.codes
    feats['ability_to_pay_estimate'] = feats['ability_to_pay_estimate'].fillna(-1.0)
    return feats

def extract_field_visit_features(field_visits: pd.DataFrame) -> pd.DataFrame:
    """Aggregates field agent dispatch history per account."""
    logger.info("Extracting field visit interaction features...")
    fv_feats = field_visits.groupby('account_id').agg(
        n_visits         = ('visit_id', 'count'),
        visits_met       = ('outcome', lambda x: x.isin(['met_borrower', 'cash_collected']).sum()),
        visits_no_trace  = ('outcome', lambda x: x.isin(['address_not_traceable', 'no_such_person']).sum()),
        avg_dwell_s      = ('dwell_s', 'mean'),
    ).reset_index()
    fv_feats['visit_success_rate'] = fv_feats['visits_met'] / fv_feats['n_visits'].clip(lower=1)
    return fv_feats

def extract_payment_features(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregates historical repayment behavior per account."""
    logger.info("Extracting repayment features...")
    pay = payments.copy()
    pay['payment_ts'] = pd.to_datetime(pay['payment_ts'])
    pay_feats = pay.groupby('account_id').agg(
        n_payments  = ('payment_id', 'count'),
        total_paid  = ('amount', 'sum'),
        avg_payment = ('amount', 'mean'),
    ).reset_index()
    return pay_feats
