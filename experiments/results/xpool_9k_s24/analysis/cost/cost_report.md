# Query-dependent pooling: accuracy vs cost (auto-generated)

dump: `/content/cv_drive/runs/xpool_9k_s24/dump_model_best/dump_original.pt`  ·  device: cuda (NVIDIA A100-SXM4-80GB, 79.3 GB)
sanity (re-computed X-Pool sims vs dump, max |diff|): 2.98e-08

## Same trained model, two scoring rules (1k-A, t2v)

| scoring | R@1 | R@5 | R@10 | MdR | MnR |
|---|---|---|---|---|---|
| X-Pool (query-dependent) | 46.7 | 71.9 | 82.0 | 2.0 | 13.8 |
| mean pooling (query-agnostic) | 42.0 | 68.9 | 79.2 | 2.0 | 17.4 |

Caveat: the encoder was trained with X-Pool pooling; mean pooling is swapped in at test time.

## Per-query top-1 outcome

- both correct 375, X-Pool only 92, mean only 45, both wrong 488
- rank improved by X-Pool 327, worse 205, same 468

## Gain of query dependence by frame_diversity_quartile

| bucket | n | R@1 mean | R@1 X-Pool | gain (pp) | median rank change |
|---|---|---|---|---|---|
| Q1 (most static) | 250 | 41.6 | 44.8 | +3.2 | +0.0 |
| Q2 | 250 | 42.4 | 48.4 | +6.0 | +0.0 |
| Q3 | 250 | 45.6 | 47.6 | +2.0 | +0.0 |
| Q4 (most diverse) | 250 | 38.4 | 46.0 | +7.6 | +0.0 |

## Gain of query dependence by query_length

| bucket | n | R@1 mean | R@1 X-Pool | gain (pp) | median rank change |
|---|---|---|---|---|---|
| <=6 | 273 | 28.2 | 33.7 | +5.5 | +0.0 |
| 7-9 | 317 | 39.1 | 41.0 | +1.9 | +0.0 |
| 10-12 | 220 | 50.5 | 54.1 | +3.6 | +0.0 |
| >=13 | 190 | 56.8 | 66.3 | +9.5 | +0.0 |

## Gain of query dependence by temporal_words

| bucket | n | R@1 mean | R@1 X-Pool | gain (pp) | median rank change |
|---|---|---|---|---|---|
| no temporal word | 937 | 41.4 | 45.7 | +4.3 | +0.0 |
| temporal word | 63 | 50.8 | 61.9 | +11.1 | +0.0 |

## Gain of query dependence by category

| bucket | n | R@1 mean | R@1 X-Pool | gain (pp) | median rank change |
|---|---|---|---|---|---|
| 0 | 74 | 39.2 | 36.5 | -2.7 | +0.0 |
| 1 | 18 | 22.2 | 27.8 | +5.6 | +0.0 |
| 2 | 53 | 22.6 | 26.4 | +3.8 | +0.0 |
| 3 | 95 | 34.7 | 40.0 | +5.3 | +0.0 |
| 4 | 58 | 46.6 | 43.1 | -3.4 | +0.0 |
| 5 | 26 | 46.2 | 53.8 | +7.7 | +0.0 |
| 6 | 30 | 30.0 | 40.0 | +10.0 | +0.0 |
| 7 | 122 | 33.6 | 41.8 | +8.2 | +0.0 |
| 8 | 38 | 42.1 | 55.3 | +13.2 | +0.0 |
| 9 | 76 | 43.4 | 50.0 | +6.6 | +0.0 |
| 10 | 46 | 41.3 | 50.0 | +8.7 | +0.0 |
| 11 | 29 | 58.6 | 62.1 | +3.4 | +0.0 |
| 12 | 43 | 51.2 | 55.8 | +4.7 | +0.0 |
| 13 | 58 | 60.3 | 62.1 | +1.7 | +0.0 |
| 14 | 47 | 44.7 | 44.7 | +0.0 | +0.0 |
| 15 | 13 | 53.8 | 69.2 | +15.4 | +0.0 |
| 16 | 64 | 48.4 | 51.6 | +3.1 | +0.0 |
| 17 | 33 | 42.4 | 48.5 | +6.1 | +0.0 |
| 18 | 53 | 47.2 | 50.9 | +3.8 | +0.0 |
| 19 | 24 | 54.2 | 62.5 | +8.3 | +0.0 |

## Two-stage retrieval (mean-pooling shortlist → X-Pool re-rank)

| P | X-Pool pairs / query | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|---|
| 1 | 1 (0.1%) | 42.0 | 68.9 | 79.2 | 17.4 |
| 5 | 5 (0.5%) | 45.6 | 68.9 | 79.2 | 17.3 |
| 10 | 10 (1.0%) | 46.3 | 71.1 | 79.2 | 17.2 |
| 20 | 20 (2.0%) | 46.7 | 72.2 | 81.5 | 16.9 |
| 50 | 50 (5.0%) | 46.6 | 72.2 | 81.8 | 16.5 |
| 100 | 100 (10.0%) | 46.8 | 72.1 | 82.0 | 16.0 |
| 200 | 200 (20.0%) | 46.7 | 71.9 | 82.0 | 15.7 |
| 500 | 500 (50.0%) | 46.7 | 71.9 | 82.0 | 15.1 |
| 1000 | 1000 (100.0%) | 46.7 | 71.9 | 82.0 | 13.8 |

## Ranking-stage cost (cuda, embeddings pre-computed, 1000 queries)

| gallery | mean ms | X-Pool ms | two-stage P=100 ms | X-Pool peak MB | index MB mean / X-Pool |
|---|---|---|---|---|---|
| 1000 | 0.23 | 96.4 | 22.7 | 1304 | 2.0 / 23.4 |
| 2000 | 0.19 | 173.9 | 24.1 | 1333 | 3.9 / 46.9 |
| 5000 | 0.36 | 433.5 | 27.3 | 1422 | 9.8 / 117.2 |
| 10000 | 0.67 | 865.9 | 32.0 | 1567 | 19.5 / 234.4 |
