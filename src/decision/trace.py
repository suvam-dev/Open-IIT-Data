import pandas as pd
import numpy as np
from typing import Tuple
from src.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger("SkipTraceOptimizer")

class SkipTraceOptimizer:
    def __init__(self, config: AppConfig):
        self.config = config

    def calculate_ev_trace(self, outstanding: float) -> float:
        """
        EV(trace) = P(find) * P(rec|found) * outstanding * collection_fraction - trace_cost
        """
        return (
            self.config.p_find_borrower_trace
            * self.config.p_recovery_given_rpc
            * outstanding
            * self.config.collection_fraction
            - self.config.avg_trace_cost_inr
        )

    def evaluate_account(self, row: pd.Series) -> Tuple[str, str]:
        """Makes account-level skip trace decision with explicit formula justification."""
        ev_t = row['ev_of_trace']
        best_prob = row['best_rpc_prob']
        out = row['outstanding']
        exhausted = row['all_contacts_exhausted']

        formula = (
            f"EV(trace)={self.config.p_find_borrower_trace:.3f}x{self.config.p_recovery_given_rpc}x"
            f"INR{out:.0f}x{self.config.collection_fraction}-INR{self.config.avg_trace_cost_inr:.0f}=INR{ev_t:.0f}"
        )

        if not exhausted and best_prob >= self.config.t_opt:
            return (
                'DO NOT TRACE',
                f"Viable contact exists P(RPC)={best_prob:.2%}>={self.config.t_opt:.2%}. {formula}"
            )
        if ev_t > 0 and exhausted:
            return (
                'TRACE',
                f"All contacts exhausted. {formula}. Net positive recovery yield."
            )
        if ev_t > 0 and best_prob < self.config.t_low:
            return (
                'TRACE',
                f"All contacts have poor connection likelihood P(RPC)<{self.config.t_low:.2%}. {formula}. Trace justified."
            )
        return (
            'DO NOT TRACE',
            f"EV(trace)=INR{ev_t:.0f} negative yield or active callable contact points available. {formula}"
        )

    def optimize_accounts(self, df_contacts: pd.DataFrame) -> pd.DataFrame:
        """Aggregates contact state to borrower account level and runs skip-trace optimization."""
        logger.info("Evaluating account-level skip-trace decisions...")
        acc_summary = df_contacts.groupby('account_id').agg(
            best_rpc_prob  = ('rpc_probability', 'max'),
            n_phones       = ('phone_id', 'count'),
            n_callable     = ('cp_action', lambda x: (x == 'Call').sum()),
            outstanding    = ('outstanding', 'first')
        ).reset_index()

        callable_phones = df_contacts[df_contacts['cp_action'] == 'Call'].groupby('account_id').first()['phone_id'].reset_index()
        callable_phones.columns = ['account_id', 'best_callable_phone']
        acc_summary = acc_summary.merge(callable_phones, on='account_id', how='left')

        acc_summary['ev_of_trace'] = acc_summary['outstanding'].apply(self.calculate_ev_trace)
        acc_summary['all_contacts_exhausted'] = (acc_summary['n_callable'] == 0)
        acc_summary['skip_trace_score'] = (1.0 - acc_summary['best_rpc_prob']) * acc_summary['outstanding']

        decisions = acc_summary.apply(self.evaluate_account, axis=1)
        acc_summary['skip_trace_decision'] = [x[0] for x in decisions]
        acc_summary['skip_trace_reasoning'] = [x[1] for x in decisions]

        logger.info(f"Account optimization complete: {len(acc_summary)} accounts processed.")
        return acc_summary
