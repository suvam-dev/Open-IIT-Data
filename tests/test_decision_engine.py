import pandas as pd
import pytest
from src.config import load_config
from src.decision.engine import DecisionEngine
from src.decision.trace import SkipTraceOptimizer

def test_compliance_overrides():
    """Asserts that third-party and invalid contacts are suppressed unconditionally."""
    config = load_config()
    engine = DecisionEngine(config)
    
    # Even with high outstanding and high RPC probability, third-party must be Suppressed
    row_tp = pd.Series({
        'is_third_party': 1,
        'is_invalid': 0,
        'rpc_probability': 0.95,
        'outstanding': 500000.0
    })
    action_tp, reason_tp = engine.recommend_contact_action(row_tp)
    assert action_tp == 'Suppress'
    assert 'third-party' in reason_tp.lower()

    # Invalid phone must also be suppressed
    row_inv = pd.Series({
        'is_third_party': 0,
        'is_invalid': 1,
        'rpc_probability': 0.90,
        'outstanding': 200000.0
    })
    action_inv, reason_inv = engine.recommend_contact_action(row_inv)
    assert action_inv == 'Suppress'
    assert 'invalid' in reason_inv.lower()

def test_ev_calculations():
    config = load_config()
    engine = DecisionEngine(config)
    
    # EV(call) = 0.5 * 0.05 * 100,000 * 0.10 - 5 = 250 - 5 = 245
    ev_c = engine.calculate_ev_call(0.5, 100000.0)
    assert pytest.approx(ev_c, rel=1e-2) == 245.0

    optimizer = SkipTraceOptimizer(config)
    # EV(trace) = 0.228 * 0.05 * 100,000 * 0.10 - 104 = 114 - 104 = 10
    ev_t = optimizer.calculate_ev_trace(100000.0)
    assert pytest.approx(ev_t, rel=1e-2) == 10.0

def test_skip_trace_justification():
    config = load_config()
    optimizer = SkipTraceOptimizer(config)
    
    # Account where all contacts are exhausted and EV(trace) > 0
    row_exhausted = pd.Series({
        'ev_of_trace': 500.0,
        'best_rpc_prob': 0.02,
        'outstanding': 500000.0,
        'all_contacts_exhausted': True
    })
    decision, reasoning = optimizer.evaluate_account(row_exhausted)
    assert decision == 'TRACE'
    
    # Account with viable active contact must NOT trace
    row_viable = pd.Series({
        'ev_of_trace': 500.0,
        'best_rpc_prob': 0.85,
        'outstanding': 500000.0,
        'all_contacts_exhausted': False
    })
    decision_v, _ = optimizer.evaluate_account(row_viable)
    assert decision_v == 'DO NOT TRACE'
