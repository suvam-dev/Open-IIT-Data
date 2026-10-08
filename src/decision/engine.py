import pandas as pd
import numpy as np
from typing import Tuple
from src.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger("DecisionEngine")

class DecisionEngine:
    def __init__(self, config: AppConfig):
        self.config = config

    def assign_health(self, p_rpc: float) -> str:
        """Assigns categorical contact health based on PR curve thresholds."""
        if p_rpc >= self.config.t_high:
            return "Excellent"
        if p_rpc >= self.config.t_opt:
            return "Good"
        if p_rpc >= self.config.t_low:
            return "Fair"
        return "Poor"

    def calculate_ev_call(self, p_rpc: float, outstanding: float) -> float:
        """
        EV(call) = P(RPC) * P(rec|RPC) * outstanding * collection_fraction - call_cost
        """
        return (
            p_rpc
            * self.config.p_recovery_given_rpc
            * outstanding
            * self.config.collection_fraction
            - self.config.call_cost_inr
        )

    def calculate_ev_visit(self, outstanding: float) -> float:
        """
        EV(visit) = P(meet) * P(rec|meet) * outstanding * collection_fraction - visit_cost
        """
        return (
            self.config.p_meet_borrower_visit
            * self.config.p_recovery_given_rpc
            * outstanding
            * self.config.collection_fraction
            - self.config.field_visit_cost_inr
        )

    def recommend_contact_action(self, row: pd.Series) -> Tuple[str, str]:
        """
        Assigns phone-level action: Call, Field Visit, or Suppress.
        Applies compliance overrides first (third-party or invalid numbers).
        """
        # 1. Compliance Hard Overrides
        if row.get('is_third_party', 0) == 1:
            return 'Suppress', 'Compliance: Verified third-party — contact strictly prohibited'
        if row.get('is_invalid', 0) == 1:
            return 'Suppress', 'Compliance: Verified invalid/wrong phone number'

        p_rpc = row['rpc_probability']
        out = row['outstanding']
        ev_c = self.calculate_ev_call(p_rpc, out)
        ev_v = self.calculate_ev_visit(out)

        # 2. Economic Optimization
        if ev_c > 0:
            return 'Call', (
                f"EV(call)=INR{ev_c:.0f} "
                f"[P(RPC)={p_rpc:.2%} x {self.config.p_recovery_given_rpc:.0%} x INR{out:.0f} "
                f"x {self.config.collection_fraction:.0%} - INR{self.config.call_cost_inr:.0f}]"
            )
        if ev_v > 0:
            return 'Field Visit', (
                f"EV(visit)=INR{ev_v:.0f} "
                f"[P(meet)={self.config.p_meet_borrower_visit:.2%} x {self.config.p_recovery_given_rpc:.0%} "
                f"x INR{out:.0f} x {self.config.collection_fraction:.0%} - INR{self.config.field_visit_cost_inr:.0f}]"
            )
        return 'Suppress', f"EV(call)=INR{ev_c:.0f}, EV(visit)=INR{ev_v:.0f} — negative economic yield"

    def process_contact_recommendations(self, df_model: pd.DataFrame) -> pd.DataFrame:
        """Applies health tiers, EV scoring, action recommendation, and account-level ranking."""
        logger.info("Generating phone-level action recommendations...")
        df = df_model.copy()
        df['contact_health'] = df['rpc_probability'].apply(self.assign_health)
        df['ev_call'] = df.apply(lambda r: self.calculate_ev_call(r['rpc_probability'], r['outstanding']), axis=1)

        actions_and_reasons = df.apply(self.recommend_contact_action, axis=1)
        df['cp_action'] = [x[0] for x in actions_and_reasons]
        df['cp_reason'] = [x[1] for x in actions_and_reasons]

        # Prioritize by account and highest RPC probability
        df = df.sort_values(['account_id', 'rpc_probability'], ascending=[True, False])
        df['cp_rank'] = df.groupby('account_id').cumcount() + 1
        return df
