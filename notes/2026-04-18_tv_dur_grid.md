# Grid search: TV + duration distribution constraint
**Date:** 2026-04-18
**Run:** `114_grid_20260418_155140` (tv ∈ {0.001, 0.05, 0.2} × dur ∈ {0.001, 0.05, 0.2, 1, 5}, 300 epochs each)

---

## Key observations

### TV stabilizes BCE holdout but destabilizes QRS holdout
- tv=0.001 → holdout BCE rises continuously to ~1.2–1.8 (no regularization)
- tv=0.05 → holdout BCE settles around ~0.5 (TV is doing real work)
- But tv=0.05/0.2 with low dur → holdout QRS diverges over training:
  - tv=0.05 dur=0.001 → ends at **87ms** holdout QRS
  - tv=0.05 dur=0.05 → **77.8ms**
  - tv=0.2 dur=0.001 → **63ms**
- Mechanism: TV smooths temporal predictions, which helps classification generalization but causes duration estimates to drift away from the true holdout distribution over long training.

### Dur constraint fixes QRS divergence but introduces train BCE spikes
- Adding dur≥1.0 at tv=0.05 → holdout QRS stabilizes (~26–28ms throughout)
- But high dur weight competes with BCE gradient → occasional spikes in **train** BCE curve (visible at tv=0.05 dur=1.0 and 5.0)
- Mechanism: dur and BCE gradients conflict; dur is pulling predictions toward training distribution while BCE is adjusting boundaries

### TV and dur are complementary — each suppresses the other's pathology
| | BCE holdout | QRS holdout |
|---|---|---|
| TV alone (high) | ✅ stable | ❌ diverges |
| Dur alone (high) | ❌ spiky train | ✅ stable |
| TV=0.05 + dur=1.0 | ✅ moderate | ✅ stable |

---

## Numerical results (best_ho / ho@best_va)

| TV | Dur | best_ho (ms) | ho@best_va (ms) | gap | notes |
|----|-----|-------------|-----------------|-----|-------|
| 0.001 | 0.001 | 24.1 | 26.7 | 2.6ms | clean curves, val less reliable |
| 0.001 | 0.2  | 23.6 | 28.4 | 4.8ms | best_ho early (ep22), val unreliable |
| 0.05  | 0.001 | 27.1 | 32.8 | 5.7ms | QRS diverges by ep300 |
| 0.05  | 0.2  | 24.9 | 34.9 | 10ms  | QRS still elevated |
| **0.05** | **1.0** | **26.5** | **27.0** | **0.5ms** | **best val/ho alignment, stable** |
| 0.05  | 5.0 | 25.8 | 27.1 | 1.3ms | stable, val slightly less tight |
| 0.2   | 0.001 | 25.7 | 29.8 | 4.1ms | QRS diverges to 63ms by end |

---

## Conclusion

**Best config: `tv=0.05, dur=1.0`**

- Holdout QRS 26.5ms, val/ho gap only 0.5ms (best_va epoch 247 ≈ best_ho epoch 246)
- Val metric is trustworthy for model selection — critical in practice
- BCE holdout well-regulated, train BCE spikes are early and settle
- tv=0.05 dur=5.0 is a close second (25.8ms, gap 1.3ms)

**Grid not fully complete** — tv=0.2 dur≥0.2 cells still running as of this writing. Worth checking if stronger TV + stronger dur holds the pattern.

---

## Open questions
- Why does TV hurt QRS holdout specifically? Possibly TV constrains the attention/boundary sharpness needed for precise duration segmentation
- The BCE train spikes with high dur: gradient clipping or LR warmup on the dur term might smooth this out
- Is the val/ho gap the right selection criterion, or should we weight best_ho more?
