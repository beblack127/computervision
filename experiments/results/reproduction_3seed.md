# X-Pool reproduction over seeds (auto-generated)

best = checkpoint selected on the test set (official protocol); last = final step. std = sample std.

## t2v · best

| run | seed | R1 | R5 | R10 | MedR | MeanR |
|---|---|---|---|---|---|---|
| xpool_9k_s24 | 24 | 46.7 | 71.9 | 82.0 | 2.0 | 13.8 |
| xpool_9k_s0 | 0 | 46.5 | 71.2 | 82.1 | 2.0 | 14.0 |
| xpool_9k_s1 | 1 | 45.3 | 73.0 | 82.0 | 2.0 | 13.1 |
| **mean ± std** | | 46.17 ± 0.76 | 72.03 ± 0.91 | 82.03 ± 0.06 | 2.00 ± 0.00 | 13.64 ± 0.45 |
| official ckpt (ours eval) | 24 | 46.9 | 72.8 | 82.2 | 2.0 | 14.3 |

## v2t · best

| run | seed | R1 | R5 | R10 | MedR | MeanR |
|---|---|---|---|---|---|---|
| xpool_9k_s24 | 24 | 45.5 | 73.1 | 83.6 | 2.0 | 8.7 |
| xpool_9k_s0 | 0 | 44.6 | 72.7 | 83.2 | 2.0 | 8.6 |
| xpool_9k_s1 | 1 | 43.8 | 72.1 | 83.9 | 2.0 | 8.4 |
| **mean ± std** | | 44.63 ± 0.85 | 72.63 ± 0.50 | 83.57 ± 0.35 | 2.00 ± 0.00 | 8.57 ± 0.15 |
| official ckpt (ours eval) | 24 | 44.4 | 73.3 | 84.0 | 2.0 | 9.0 |

## t2v · last

| run | seed | R1 | R5 | R10 | MedR | MeanR |
|---|---|---|---|---|---|---|
| xpool_9k_s24 | 24 | 46.7 | 71.9 | 82.0 | 2.0 | 13.8 |
| xpool_9k_s0 | 0 | 46.0 | 71.1 | 81.9 | 2.0 | 14.0 |
| xpool_9k_s1 | 1 | 44.7 | 72.8 | 82.2 | 2.0 | 13.2 |
| **mean ± std** | | 45.80 ± 1.01 | 71.93 ± 0.85 | 82.03 ± 0.15 | 2.00 ± 0.00 | 13.66 ± 0.39 |
| official ckpt (ours eval) | 24 | 46.9 | 72.8 | 82.2 | 2.0 | 14.3 |

## v2t · last

| run | seed | R1 | R5 | R10 | MedR | MeanR |
|---|---|---|---|---|---|---|
| xpool_9k_s24 | 24 | 45.5 | 73.1 | 83.6 | 2.0 | 8.7 |
| xpool_9k_s0 | 0 | 44.2 | 72.6 | 83.2 | 2.0 | 8.6 |
| xpool_9k_s1 | 1 | 43.9 | 72.0 | 83.7 | 2.0 | 8.3 |
| **mean ± std** | | 44.53 ± 0.85 | 72.57 ± 0.55 | 83.50 ± 0.26 | 2.00 ± 0.00 | 8.54 ± 0.18 |
| official ckpt (ours eval) | 24 | 44.4 | 73.3 | 84.0 | 2.0 | 9.0 |

best − last (t2v R@1): [0.0, 0.5, 0.6] → mean 0.37

## Cost per run

| run | GPU | sessions | wall h | peak reserved GB | s/step (median) | best step |
|---|---|---|---|---|---|---|
| xpool_9k_s24 | NVIDIA A100-SXM4-80GB | 3 | 7.27 (≥) * | 17.56 | 0.689 | 28125 |
| xpool_9k_s0 | NVIDIA A100-SXM4-40GB | 1 | 6.128 | 17.66 | 0.703 | 25875 |
| xpool_9k_s1 | NVIDIA A100-SXM4-40GB | 1 | 6.112 | 17.75 | 0.702 | 24188 |

\* reconstructed lower bound: run_meta of that run covered only its last session (sessions_reconstructed.json)
