# Results — 2026-04-18

All runs use `--augment` where noted. QT ignored (always 0.0). Metric: QRS MAE (ms), lower is better.

| Run | Script | Augment | Epochs | best_va (epoch) | ho@best_va | best_ho (epoch) | va@best_ho |
|-----|--------|---------|--------|----------------|------------|----------------|------------|
| 113 short | train_semi | yes | 300  | 8.1 @ 139  | 38.5 | 29.1 @ 5   | 22.3 |
| 113 long  | train_semi | yes | 10000 | 7.3 @ 2923 | 38.1 | 26.1 @ 23  | 13.2 |
| 114 short | train      | yes | 300  | 5.8 @ 177  | 27.1 | 22.4 @ 91  | 6.6  |
| 114 long  | train      | yes | 10000 | 5.8 @ 4596 | 27.5 | **21.3 @ 4937** | 6.2 |
| 115 short | train      | no  | 300  | 15.4 @ 30  | 32.3 | 26.2 @ 19  | 20.1 |
| 115 long  | train      | no  | 10000 | 14.8 @ 200 | 32.9 | 29.4 @ 2121 | 17.4 |

## Winner: 114 long (train + augment, 10000 epochs)
- best_va = 5.8ms, best_ho = 21.3ms
- Augmentation is essential (~6ms va vs ~15ms without)
- Semi-supervised (113) hurts holdout despite similar val loss (38ms ho vs 27ms)
