# CreditNirvana PS2 — Right-Party Contact Prediction & Skip-Trace Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-HistGradBoost%20%7C%20RandomForest-green.svg)](https://scikit-learn.org/)
[![Testing](https://img.shields.io/badge/pytest-9%20passed-success.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20Production-blueviolet.svg)](#)

Production-grade Machine Learning repository for **Problem Statement 2 (PS2)**: Predicting Right-Party Contact (RPC) probabilities per contact point and establishing an economically sound skip-trace prioritization engine for delinquent accounts.

---

## 1. Executive Summary & Problem Framing

Collections operations face two primary inefficiencies:
1. **Low Contact Yield:** Dialing unresponsive or incorrect phone numbers burns collector bandwidth and incurs unnecessary telecom costs.
2. **Expensive Skip-Tracing:** External skip-trace investigations cost **INR 104 per request** with a **22.8% historical success rate**. Tracing must only be triggered when primary contacts are exhausted and expected debt recovery exceeds vendor costs.

### Two-Tier Decision Architecture
```
                         [Borrower Account]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[Contact Point 1]                                 [Contact Point 2] ...
         │                                                 │
 [Feature Store]                                   [Feature Store]
         │                                                 │
[Calibrated ML Model]                             [Calibrated ML Model]
         │                                                 │
 Calibrated P(RPC)                                 Calibrated P(RPC)
         │                                                 │
   [Tier 1 Engine]                                   [Tier 1 Engine]
(Call / Visit / Suppress)                         (Call / Visit / Suppress)
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  │
                    Are primary contacts exhausted?
                                  │
                  ┌───────────────┴───────────────┐
                  ▼ YES                           ▼ NO
           EV(trace) > 0?                     DO NOT TRACE
           ┌──────┴──────┐                 (Call viable phone)
           ▼ YES         ▼ NO
         TRACE       DO NOT TRACE
     (Economical)    (Negative EV)
```

---

## 2. Repository Architecture & Layout

```
ps2_right_party_contact/
├── configs/
│   ├── config.yaml                    # Environment paths, business parameters, calibration settings
│   └── model_params.yaml              # Hyperparameter grids & candidate model architectures
├── src/
│   ├── config.py                      # Strongly-typed configuration parser
│   ├── data/
│   │   ├── loader.py                  # Validated ingestion with schema assertions
│   │   └── schema.py                  # Table contracts & target label formulations
│   ├── features/
│   │   ├── telephony.py               # Leakage-safe call recency & failure streak features
│   │   ├── account.py                 # Borrower risk & omnichannel interaction aggregators
│   │   └── pipeline.py                # Consolidated feature store assembler
│   ├── models/
│   │   ├── train.py                   # 5-fold Stratified CV & candidate training
│   │   ├── tune.py                    # Cross-validated hyperparameter optimization
│   │   ├── calibration.py             # Isotonic regression + Bayesian dummy-weight smoothing
│   │   ├── evaluate.py                # ROC AUC, Brier score, PR curves, and plotting
│   │   └── registry.py                # Joblib model serialization & loading
│   ├── decision/
│   │   ├── engine.py                  # Phone-level EV equations, health tiers & compliance
│   │   └── trace.py                   # Account-level skip-trace optimizer & justification engine
│   └── utils/
│       └── logger.py                  # Structured logging
├── tests/
│   ├── test_data_loader.py            # Data ingestion contracts & schema tests
│   ├── test_feature_leakage.py        # Automated test proving ZERO target leakage
│   ├── test_calibration.py            # Verification of probability calibration bounds
│   └── test_decision_engine.py        # Business EV calculations & compliance assertions
├── artifacts/
│   └── models/
│       └── champion_model.joblib      # Persisted production model bundle
├── outputs/
│   ├── ps2_contact_recommendations.csv# 5,719 contact-level decisions & rankings
│   ├── ps2_account_decisions.csv      # 2,400 account-level skip-trace recommendations
│   └── model_comparison.csv          # Candidate benchmark evaluation table
├── reports/
│   └── model_evaluation.png           # ROC, Calibration, and Precision-Recall plots
├── creditnirvana_ps2.ipynb            # Interactive, reproducible presentation notebook
├── main.py                            # Production CLI orchestrator (train, predict, pipeline)
├── run_ml.py                          # Backwards-compatible execution entrypoint
└── requirements.txt                   # Production package dependencies
```

---

## 3. Machine Learning Innovations

### A. Zero Target Leakage Feature Pipeline
Previous naive implementations suffered from target leakage by including historical RPC counts/dispositions in the feature set. Our feature pipeline strictly excludes all target derivatives:
* **Excluded:** `has_rpc`, `rpc_count`, `rpc_rate`, `verified_rpc`, `has_ptp`.
* **Retained (Observable Signals):** Dial frequency (`total_attempts`, `attempts_7d`), ring and talk durations (`avg_ring_s`, `avg_talk_s`, `pct_zero_talk`), failure streaks (`recent_failures`, `switched_off_count`, `wrong_number_count`), and borrower account metadata.

### B. Isotonic Probability Calibration with Bayesian Smoothing
Raw gradient-boosted trees produce extreme, uncalibrated log-odds. We apply:
1. **Isotonic Calibration:** Fitted on a held-out validation fold (`validation`), aligning predicted probabilities with empirical frequencies.
2. **Bayesian Dummy-Weight Smoothing:**
   $$P_{\text{smoothed}} = \frac{P_{\text{calibrated}} + w_{\text{dummy}} \times P_{\text{prior}}}{1 + w_{\text{dummy}}}$$
   With $w_{\text{dummy}} = 0.02$ and population baseline $P_{\text{prior}} \approx 0.448$, probabilities never collapse to hard `0.000000` or `1.000000`. Unreachable phones settle at a realistic floor (~0.88%), preserving valid expected value calculations.

### C. Data-Driven Health Tiers
Derived from the validation fold's Precision-Recall curve:
* **Excellent** ($P \ge 0.70$): High confidence, precision > 95%.
* **Good** ($0.40 \le P < 0.70$): Balanced operational tier.
* **Fair** ($0.15 \le P < 0.40$): Marginal callable contact points.
* **Poor** ($P < 0.15$): Low contact probability.

---

## 4. Benchmark Evaluation Results

Evaluated on the official out-of-time test set (837 contact points):

| Model | 5-Fold CV ROC-AUC | Test AUC (Calibrated) | Test Brier Score (Calibrated) | Status |
|---|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | **0.9797 ± 0.0042** | **0.9780** | **0.0492** | **Champion (Persisted)** |
| Random Forest | 0.9786 ± 0.0048 | 0.9756 | 0.0524 | Candidate |
| Logistic Regression | 0.9174 ± 0.0120 | 0.9024 | 0.1141 | Baseline |

---

## 5. Decision Layer: Expected Value Engine

All formulas are fully transparent with parameters categorized as `[DATA-DERIVED]` or `[ASSUMED]`:

$$\text{EV}(\text{Call}) = P(\text{RPC}) \times P(\text{Rec}|\text{RPC}) \times \text{Outstanding} \times \text{ColFrac} - \text{Cost}_{\text{call}}$$
$$\text{EV}(\text{Visit}) = P(\text{Meet}) \times P(\text{Rec}|\text{Met}) \times \text{Outstanding} \times \text{ColFrac} - \text{Cost}_{\text{visit}}$$
$$\text{EV}(\text{Trace}) = P(\text{Find}) \times P(\text{Rec}|\text{Found}) \times \text{Outstanding} \times \text{ColFrac} - \text{Cost}_{\text{trace}}$$

* $P(\text{Find via Trace}) = 0.228$ `[DATA-DERIVED: 175 / 766 historical traces]`
* $P(\text{Meet via Visit}) = 0.216$ `[DATA-DERIVED: 1,206 / 5,578 field visits]`
* $\text{Cost}_{\text{trace}} = \text{INR } 104$ `[DATA-DERIVED: Average skip trace vendor cost]`
* $P(\text{Recovery}|\text{Contact}) = 0.05$ `[ASSUMED: 5% collection likelihood upon reach]`
* $\text{Collection Fraction} = 0.10$ `[ASSUMED: 10% average recovered balance]`
* $\text{Cost}_{\text{call}} = \text{INR } 5$, $\text{Cost}_{\text{visit}} = \text{INR } 200$ `[ASSUMED: Operations standard]`

---

## 6. How to Run, Test & Reproduce

### 1. Execute Unit & Integration Tests
```bash
pytest -v
```
Runs 9 comprehensive automated tests validating schema constraints, target generation, zero-leakage assertions, calibration monotonicity, and decision engine logic.

### 2. Run End-to-End Pipeline via CLI
```bash
# Complete pipeline (ingest -> train -> calibrate -> persist -> predict -> export)
python main.py pipeline

# Run training only
python main.py train

# Run batch inference from persisted champion model artifact
python main.py predict
```

### 3. Interactive Jupyter Presentation
```bash
jupyter notebook creditnirvana_ps2.ipynb
```
Executes the modular pipeline with inline visualizations, calibration charts, and business simulation tables.
