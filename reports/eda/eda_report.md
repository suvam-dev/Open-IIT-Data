# Exploratory Data Analysis Report

## 1. Data Overview
- **Number of rows:** 5719
- **Number of columns:** 56

## 2. Missing Values
|                          |   Missing Count |
|:-------------------------|----------------:|
| avg_trace_cost           |            4261 |
| total_trace_cost         |            4261 |
| traces_successful        |            4261 |
| n_traces                 |            4261 |
| salary_credit_day        |            3831 |
| visit_success_rate       |            2537 |
| avg_dwell_s              |            2537 |
| visits_no_trace          |            2537 |
| visits_met               |            2537 |
| n_visits                 |            2537 |
| ability_to_pay_estimate  |            1760 |
| avg_payment              |            1727 |
| total_paid               |            1727 |
| n_payments               |            1727 |
| failure_rate             |            1639 |
| talk_rate                |            1639 |
| total_attempts           |            1639 |
| wrong_number_count       |            1639 |
| attempts_30d             |            1639 |
| attempts_7d              |            1639 |
| days_since_last_attempt  |            1639 |
| days_since_first_attempt |            1639 |
| n_channels               |            1639 |
| n_agents                 |            1639 |
| pct_zero_ring            |            1639 |
| pct_zero_talk            |            1639 |
| max_talk_s               |            1639 |
| avg_talk_s               |            1639 |
| avg_ring_s               |            1639 |
| invalid_number_count     |            1639 |
| not_reachable_count      |            1639 |
| switched_off_count       |            1639 |
| recent_failures          |            1639 |

## 3. Data Types
|                          | Data Type   |
|:-------------------------|:------------|
| phone_id                 | str         |
| account_id               | str         |
| target_rpc               | int64       |
| split                    | str         |
| source                   | str         |
| priority_slot            | int64       |
| added_date               | str         |
| verified_status          | str         |
| is_third_party           | int64       |
| is_invalid               | int64       |
| total_attempts           | float64     |
| wrong_number_count       | float64     |
| switched_off_count       | float64     |
| not_reachable_count      | float64     |
| invalid_number_count     | float64     |
| avg_ring_s               | float64     |
| avg_talk_s               | float64     |
| max_talk_s               | float64     |
| pct_zero_talk            | float64     |
| pct_zero_ring            | float64     |
| n_agents                 | float64     |
| n_channels               | float64     |
| days_since_first_attempt | float64     |
| days_since_last_attempt  | float64     |
| attempts_7d              | float64     |
| attempts_30d             | float64     |
| recent_failures          | float64     |
| failure_rate             | float64     |
| talk_rate                | float64     |
| outstanding              | float64     |
| dpd_start                | int64       |
| emi_amount               | float64     |
| bureau_score_band        | str         |
| other_active_loans       | int64       |
| prev_ptp_count           | int64       |
| prev_ptp_broken          | int64       |
| ability_to_pay_estimate  | float64     |
| income_type              | str         |
| dialling_arm             | str         |
| salary_credit_day        | float64     |
| lender_id                | str         |
| lender_name              | str         |
| lender_type              | str         |
| phones_per_account       | int64       |
| n_visits                 | float64     |
| visits_met               | float64     |
| visits_no_trace          | float64     |
| avg_dwell_s              | float64     |
| visit_success_rate       | float64     |
| n_payments               | float64     |
| total_paid               | float64     |
| avg_payment              | float64     |
| n_traces                 | float64     |
| traces_successful        | float64     |
| total_trace_cost         | float64     |
| avg_trace_cost           | float64     |

## 4. Descriptive Statistics (Numerical)
|                          |   count |            mean |            std |        min |          25% |            50% |           75% |            max |
|:-------------------------|--------:|----------------:|---------------:|-----------:|-------------:|---------------:|--------------:|---------------:|
| target_rpc               |    5719 |      0.447806   |      0.497312  |    0       |     0        |      0         |      1        |      1         |
| priority_slot            |    5719 |      0.86676    |      0.898336  |    0       |     0        |      1         |      1        |      5         |
| is_third_party           |    5719 |      0.0150376  |      0.121713  |    0       |     0        |      0         |      0        |      1         |
| is_invalid               |    5719 |      0.00856793 |      0.0921738 |    0       |     0        |      0         |      0        |      1         |
| total_attempts           |    4080 |     12.835      |      8.66893   |    1       |     7        |     10         |     18        |     57         |
| wrong_number_count       |    4080 |      0.303186   |      0.636774  |    0       |     0        |      0         |      0        |      3         |
| switched_off_count       |    4080 |      0.817647   |      2.87887   |    0       |     0        |      0         |      0        |     39         |
| not_reachable_count      |    4080 |      1.09632    |      1.80674   |    0       |     0        |      1         |      2        |     26         |
| invalid_number_count     |    4080 |      0.0215686  |      0.206597  |    0       |     0        |      0         |      0        |      2         |
| avg_ring_s               |    4080 |     20.3226     |      8.37952   |    0       |    17        |     21.8844    |     25.7157   |     45         |
| avg_talk_s               |    4080 |     30.9097     |     31.1671    |    0       |     5.77083  |     25.1292    |     47.2708   |    406         |
| max_talk_s               |    4080 |    155.638      |    142.182     |    0       |    31        |    125         |    296        |    420         |
| pct_zero_talk            |    4080 |      0.700491   |      0.244052  |    0       |     0.571429 |      0.733333  |      0.882353 |      1         |
| pct_zero_ring            |    4080 |      0.15521    |      0.269742  |    0       |     0        |      0.0563492 |      0.142857 |      1         |
| n_agents                 |    4080 |      1.24706    |      0.456213  |    1       |     1        |      1         |      1        |      4         |
| n_channels               |    4080 |      1.22892    |      0.42019   |    1       |     1        |      1         |      1        |      2         |
| days_since_first_attempt |    4080 |     72.5085     |     21.6927    |    0.14478 |    62.1405   |     83.4469    |     88.308    |     89.4539    |
| days_since_last_attempt  |    4080 |     32.8086     |     28.2432    |    0       |     5.35799  |     26.1527    |     58.0812   |     89.448     |
| attempts_7d              |    4080 |      0.666176   |      1.28595   |    0       |     0        |      0         |      1        |      8         |
| attempts_30d             |    4080 |      3.26985    |      4.15842   |    0       |     0        |      1         |      6        |     24         |
| recent_failures          |    4080 |      0.845833   |      1.36697   |    0       |     0        |      0         |      1        |      5         |
| failure_rate             |    4080 |      0.19152    |      0.280753  |    0       |     0        |      0.0909091 |      0.222222 |      1         |
| talk_rate                |    4080 |      0.827696   |      0.377691  |    0       |     1        |      1         |      1        |      1         |
| outstanding              |    5719 | 216699          | 252059         | 6300       | 62500        | 126500         | 267900        |      2.117e+06 |
| dpd_start                |    5719 |     65.6718     |     86.2813    |    0       |    11        |     41         |     79        |    720         |
| emi_amount               |    5719 |   7512.43       |   7428.57      |  700       |  2650        |   4400         |  10200        |  39950         |
| other_active_loans       |    5719 |      1.90208    |      1.38795   |    0       |     1        |      2         |      3        |      8         |
| prev_ptp_count           |    5719 |      1.54293    |      1.57205   |    0       |     0        |      1         |      2        |     10         |
| prev_ptp_broken          |    5719 |      0.812729   |      1.09838   |    0       |     0        |      0         |      1        |      7         |
| ability_to_pay_estimate  |    3959 |      0.53339    |      0.245525  |    0       |     0.36     |      0.54      |      0.71     |      1         |
| salary_credit_day        |    1888 |     11.1028     |     10.0266    |    1       |     1        |      7         |     15        |     30         |
| phones_per_account       |    5719 |      2.73352    |      0.89594   |    1       |     2        |      3         |      3        |      6         |
| n_visits                 |    3182 |      4.14708    |      2.73124   |    1       |     2        |      4         |      7        |      9         |
| visits_met               |    3182 |      0.877121   |      1.22621   |    0       |     0        |      0         |      1        |      8         |
| visits_no_trace          |    3182 |      1.1939     |      1.51295   |    0       |     0        |      1         |      2        |      9         |
| avg_dwell_s              |    3182 |    340.323      |    253.465     |   30       |   154.5      |    267.583     |    473.333    |   1481         |
| visit_success_rate       |    3182 |      0.213975   |      0.296937  |    0       |     0        |      0         |      0.333333 |      1         |
| n_payments               |    3992 |      1.29259    |      0.576031  |    1       |     1        |      1         |      1        |      6         |
| total_paid               |    3992 |  16203          |  29293         |  300       |  3300        |   7400         |  17000        | 438000         |
| avg_payment              |    3992 |  14194.6        |  27278.8       |  233.333   |  2600        |   6200         |  14200        | 438000         |
| n_traces                 |    1458 |      1.25583    |      0.46095   |    1       |     1        |      1         |      1        |      3         |
| traces_successful        |    1458 |      0.341564   |      0.482999  |    0       |     0        |      0         |      1        |      2         |
| total_trace_cost         |    1458 |    130.474      |     56.6157    |   60       |    90.25     |    118         |    148        |    429         |
| avg_trace_cost           |    1458 |    103.997      |     25.0747    |   60       |    83        |    102         |    124        |    150         |

## 5. Descriptive Statistics (Categorical)
|                   |   count |   unique | top             |   freq |
|:------------------|--------:|---------:|:----------------|-------:|
| phone_id          |    5719 |     5618 | PH000546        |      5 |
| account_id        |    5719 |     2400 | AC000057        |      6 |
| split             |    5719 |        3 | train           |   4001 |
| source            |    5719 |        6 | kyc_origination |   2400 |
| added_date        |    5719 |       63 | 2026-04-01      |   5571 |
| verified_status   |    5719 |        6 | unverified      |   5457 |
| bureau_score_band |    5719 |        5 | 300-549         |   3566 |
| income_type       |    5719 |        3 | salaried        |   3205 |
| dialling_arm      |    5719 |        2 | rule_based      |   5448 |
| lender_id         |    5719 |        6 | L01             |   1592 |
| lender_name       |    5719 |        6 | Lender A        |   1592 |
| lender_type       |    5719 |        6 | Private bank    |   1592 |

