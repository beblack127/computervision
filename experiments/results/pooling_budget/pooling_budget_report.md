# Query-dependent pooling on a budget (auto-generated, 3 runs: xpool_9k_s24, xpool_9k_s0, xpool_9k_s1)

mean ± sample std over runs. U-Pool = trained X-Pool head with uniform attention (query-agnostic, pre-computable). budget = X-Pool pair evaluations per query.

## 1. Where does X-Pool's gain come from? (same trained model, three scoring rules)

| scoring | query-dependent? | index per video | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|---|---|
| mean | no | 1 vector | 41.8 ± 0.6 | 68.5 ± 0.5 | 78.8 ± 0.8 | 17.4 ± 0.3 |
| upool | no | 1 vector | 43.2 ± 0.2 | 68.8 ± 0.7 | 79.4 ± 1.0 | 15.8 ± 0.4 |
| xpool | yes | 12 frame vectors | 46.2 ± 0.8 | 72.0 ± 0.9 | 82.0 ± 0.1 | 13.6 ± 0.5 |

| top-1 transitions (count of 1000 queries) | mean ± std |
|---|---|
| correct_mean | 417.7 ± 5.9 |
| correct_upool | 431.7 ± 2.3 |
| correct_xpool | 461.7 ± 7.6 |
| fixed_by_projection(mean wrong, upool right) | 49.3 ± 2.3 |
| broken_by_projection(mean right, upool wrong) | 35.3 ± 5.5 |
| fixed_by_attention(upool wrong, xpool right) | 67.0 ± 3.6 |
| broken_by_attention(upool right, xpool wrong) | 37.0 ± 1.7 |
| oracle_any_of_three | 520.0 ± 7.0 |

## 2. Attention sharpness (logits × alpha; 0 = uniform = U-Pool, 1 = X-Pool)

| alpha | R@1 | R@5 | R@10 |
|---|---|---|---|
| 0.0 | 43.2 ± 0.2 | 68.8 ± 0.7 | 79.4 ± 1.0 |
| 0.25 | 44.5 ± 0.1 | 70.2 ± 0.9 | 80.1 ± 0.5 |
| 0.5 | 45.8 ± 0.5 | 70.9 ± 0.7 | 81.2 ± 0.3 |
| 0.75 | 46.1 ± 0.9 | 71.6 ± 0.8 | 81.6 ± 0.3 |
| 1.0 | 46.2 ± 0.8 | 72.0 ± 0.9 | 82.0 ± 0.1 |
| 1.5 | 45.0 ± 0.8 | 72.3 ± 0.7 | 82.4 ± 0.2 |
| 2.0 | 44.0 ± 0.5 | 71.7 ± 0.7 | 82.2 ± 0.4 |

## 3. Hubness (k=10)

| scoring | N_k skewness | max N1 | videos never top-1 |
|---|---|---|---|
| mean | 1.04 ± 0.06 | 12.3 ± 3.8 | 383.7 ± 3.1 |
| upool | 0.94 ± 0.05 | 8.0 ± 1.7 | 368.7 ± 7.4 |
| xpool | 0.89 ± 0.01 | 7.0 ± 1.0 | 348.3 ± 11.7 |
| xpool + QB-Norm | 0.55 ± 0.08 | 7.3 ± 0.6 | 357.3 ± 8.1 |

## 4. Two-stage retrieval: accuracy vs X-Pool budget

E1 = mean→X-Pool fixed P (X-Pool App. C; Bain et al. 2022) · E2 = mean→X-Pool adaptive depth (ranked-list truncation, adapted) · O1/O2 = U-Pool first stage (own).

### E1_mean_fixed

| P / target | avg pairs/query | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|---|
| 1 | 1.0 | 41.8 ± 0.6 | 68.5 ± 0.5 | 78.8 ± 0.8 | 17.4 ± 0.3 |
| 2 | 2.0 | 44.4 ± 0.6 | 68.5 ± 0.5 | 78.8 ± 0.8 | 17.4 ± 0.3 |
| 3 | 3.0 | 44.7 ± 0.2 | 68.5 ± 0.5 | 78.8 ± 0.8 | 17.4 ± 0.3 |
| 5 | 5.0 | 45.2 ± 0.6 | 68.5 ± 0.5 | 78.8 ± 0.8 | 17.3 ± 0.3 |
| 10 | 10.0 | 45.9 ± 0.8 | 71.0 ± 0.7 | 78.8 ± 0.8 | 17.2 ± 0.3 |
| 20 | 20.0 | 46.2 ± 1.0 | 72.2 ± 0.7 | 81.5 ± 0.2 | 16.9 ± 0.4 |
| 50 | 50.0 | 46.1 ± 0.9 | 72.0 ± 1.0 | 81.8 ± 0.2 | 16.6 ± 0.4 |
| 100 | 100.0 | 46.2 ± 0.8 | 72.1 ± 1.0 | 82.0 ± 0.0 | 15.9 ± 0.4 |

### O1_upool_fixed

| P / target | avg pairs/query | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|---|
| 1 | 1.0 | 43.2 ± 0.2 | 68.8 ± 0.7 | 79.4 ± 1.0 | 15.8 ± 0.4 |
| 2 | 2.0 | 44.6 ± 0.4 | 68.8 ± 0.7 | 79.4 ± 1.0 | 15.8 ± 0.4 |
| 3 | 3.0 | 45.1 ± 0.3 | 68.8 ± 0.7 | 79.4 ± 1.0 | 15.8 ± 0.4 |
| 5 | 5.0 | 45.4 ± 0.6 | 68.8 ± 0.7 | 79.4 ± 1.0 | 15.7 ± 0.4 |
| 10 | 10.0 | 45.8 ± 1.0 | 70.9 ± 0.6 | 79.4 ± 1.0 | 15.6 ± 0.3 |
| 20 | 20.0 | 46.2 ± 1.0 | 71.9 ± 0.6 | 81.0 ± 0.3 | 15.5 ± 0.4 |
| 50 | 50.0 | 46.2 ± 0.9 | 72.0 ± 0.9 | 81.7 ± 0.1 | 15.2 ± 0.4 |
| 100 | 100.0 | 46.2 ± 0.9 | 71.9 ± 0.8 | 81.8 ± 0.1 | 14.7 ± 0.4 |

### E2_mean_adaptive_P0-50

| P / target | avg pairs/query | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|---|
| 2 | 2.0 | 41.9 ± 0.4 | 68.6 ± 0.4 | 78.8 ± 0.7 | 17.4 ± 0.3 |
| 3 | 3.0 | 42.0 ± 0.6 | 68.6 ± 0.5 | 78.8 ± 0.6 | 17.4 ± 0.3 |
| 5 | 5.0 | 42.2 ± 0.8 | 68.7 ± 0.7 | 79.2 ± 0.5 | 17.3 ± 0.4 |
| 10 | 10.0 | 43.2 ± 1.2 | 69.2 ± 0.6 | 79.6 ± 0.4 | 17.3 ± 0.3 |
| 20 | 20.0 | 44.4 ± 1.3 | 69.9 ± 0.9 | 80.4 ± 0.4 | 17.0 ± 0.3 |

### O2_upool_adaptive_P0-50

| P / target | avg pairs/query | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|---|
| 2 | 2.0 | 43.3 ± 0.5 | 69.1 ± 0.8 | 79.7 ± 0.9 | 15.8 ± 0.4 |
| 3 | 3.0 | 43.4 ± 0.4 | 69.4 ± 0.9 | 79.7 ± 0.7 | 15.8 ± 0.4 |
| 5 | 5.0 | 43.5 ± 0.2 | 69.6 ± 1.1 | 79.9 ± 0.5 | 15.7 ± 0.4 |
| 10 | 10.0 | 44.0 ± 0.5 | 70.0 ± 0.8 | 79.9 ± 0.6 | 15.7 ± 0.4 |
| 20 | 20.0 | 44.7 ± 0.9 | 70.6 ± 0.9 | 80.3 ± 0.5 | 15.5 ± 0.4 |

### E2_mean_adaptive_P5-50

| P / target | avg pairs/query | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|---|
| 5 | 5.0 | 45.2 ± 0.6 | 68.5 ± 0.5 | 78.8 ± 0.8 | 17.3 ± 0.3 |
| 10 | 10.0 | 45.3 ± 0.7 | 68.8 ± 0.8 | 79.2 ± 0.4 | 17.2 ± 0.4 |
| 20 | 20.0 | 45.5 ± 0.9 | 69.6 ± 0.8 | 80.1 ± 0.2 | 17.1 ± 0.3 |

### O2_upool_adaptive_P5-50

| P / target | avg pairs/query | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|---|
| 5 | 5.0 | 45.4 ± 0.6 | 68.8 ± 0.7 | 79.4 ± 1.0 | 15.7 ± 0.4 |
| 10 | 10.0 | 45.5 ± 0.6 | 69.6 ± 1.0 | 79.9 ± 0.4 | 15.7 ± 0.4 |
| 20 | 20.0 | 45.8 ± 0.6 | 70.5 ± 1.0 | 80.3 ± 0.4 | 15.5 ± 0.4 |

## 6. Accuracy without training (λ = 1 and β = 20 fixed a priori; other values = sensitivity only)

| method | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|
| xpool | 46.2 ± 0.8 | 72.0 ± 0.9 | 82.0 ± 0.1 | 13.6 ± 0.5 |
| fusion xpool+0.25*upool | 46.0 ± 0.8 | 71.9 ± 0.8 | 81.8 ± 0.2 | 13.8 ± 0.4 |
| fusion xpool+0.25*mean | 46.2 ± 1.1 | 72.1 ± 0.8 | 81.5 ± 0.2 | 13.8 ± 0.4 |
| fusion xpool+0.5*upool | 45.9 ± 0.7 | 71.6 ± 0.9 | 81.6 ± 0.2 | 13.9 ± 0.4 |
| fusion xpool+0.5*mean | 46.1 ± 1.4 | 72.5 ± 0.4 | 81.4 ± 0.2 | 14.1 ± 0.4 |
| fusion xpool+1.0*upool | 45.3 ± 0.4 | 71.0 ± 0.4 | 81.1 ± 0.4 | 14.2 ± 0.4 |
| fusion xpool+1.0*mean | 45.1 ± 0.6 | 71.7 ± 0.3 | 81.3 ± 0.1 | 14.6 ± 0.4 |
| fusion xpool+2.0*upool | 44.7 ± 0.4 | 70.4 ± 0.6 | 80.3 ± 0.4 | 14.6 ± 0.4 |
| fusion xpool+2.0*mean | 44.5 ± 0.8 | 71.0 ± 0.6 | 80.7 ± 0.4 | 15.2 ± 0.4 |
| xpool + QB-Norm | 45.2 ± 0.9 | 71.1 ± 0.5 | 82.1 ± 0.4 | 13.7 ± 0.3 |
| mean + QB-Norm | 41.0 ± 0.4 | 66.9 ± 0.8 | 77.9 ± 0.3 | 17.3 ± 0.3 |
| upool + QB-Norm | 41.8 ± 0.3 | 68.0 ± 0.4 | 79.3 ± 0.4 | 16.1 ± 0.1 |
| fusion xpool+1.0*upool + QB-Norm | 44.5 ± 0.3 | 70.9 ± 0.8 | 82.3 ± 0.3 | 13.8 ± 0.3 |
| xpool + QB-Norm (beta=10.0) | 44.8 ± 0.4 | 70.8 ± 0.4 | 81.5 ± 0.9 | 13.9 ± 0.3 |
| xpool + QB-Norm (beta=40.0) | 45.9 ± 0.5 | 72.2 ± 1.0 | 83.2 ± 0.5 | 13.2 ± 0.3 |

## 5. Ranking-stage time (first run, NVIDIA A100-SXM4-80GB, 1000 queries, ms)

| gallery | upool_index_build_ms_offline | first_stage_ms | xpool_all_pairs_ms | two_stage_P5_ms | two_stage_P20_ms | adaptive_B5_P0-50_ms |
|---|---|---|---|---|---|---|
| 1000 | 1.05 | 0.25 | 87.26 | 6.33 | 6.43 | 3.89 |
| 10000 | 8.48 | 1.08 | 866.47 | 10.58 | 14.11 | 42.93 |

