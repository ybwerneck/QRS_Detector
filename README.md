# QRS Duration Detector — Phase 0 Checkpoint

ECG QRS and QT interval regression using a frozen HuBERT-ECG-base encoder.

## Architecture

- **Frozen HuBERT-ECG-base encoder** — no gradient updates
- **LearnedCompressor** (conv, ~490K params) — compresses encoder output across leads and time
- **decision_enc** MLP (550→512→256→256→64, ~496K params) — injects temporal context via Pan-Tompkins decision signal
- **qrs_head** (128→640→640→1, ~493K params) — scalar QRS duration output
- **qt_head** (129→640→640→1, ~494K params) — scalar QT duration output (conditioned on QRS prediction)

**Total trainable params: ~1.97M**

## Phase 0 Results

- Train error: below clinical noise floor (~10 ms)
- Same-patient validation: ~10 ms
- New-patient holdout: degrades (expected — patient generalization deferred to Phase 1/2)

**Conclusion:** representational capacity confirmed; decision signal carries temporal structure; ready for logit head.

---

# Phase 2 — Logit Mask Head

Replaced scalar regression with a **soft mask** over the beat window: the model now predicts a per-ms probability and the duration is the sum of the mask.

## Architecture

- **g path (compressor):** `(N, 12, t, 768)` → depthwise Conv2d stack with learned lead-collapse → `(N, 1, 550)`, z-scored
- **f path (PT anchor):** z-scored Pan-Tompkins decision signal `(N, 1, 550)` — directly anchors the mask to beat structure
- **Fusion:** `cat[g, f]` → Conv1d × 3 → logits `(N, 1, 550)` → sigmoid → mask; duration = mask.sum()
- **Loss:** BCE on ground-truth rectangular mask + MAE regularisation

## Phase 2 Results (epoch 10 000)

| Split | QRS MAE | QT MAE |
|---|---|---|
| Train | 0.0 ms | 0.0 ms |
| Val | 17.9 ms | 0.0 ms |
| Holdout (p9) | 22.9 ms | 0.0 ms |

**Conclusion:** mask head achieves near-zero train error; PT anchor provides strong temporal signal; val/holdout generalisation competitive with Phase 0 scalar head.
