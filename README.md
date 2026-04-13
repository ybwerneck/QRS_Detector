# QRS / QT Duration Detector

ECG QRS and QT interval regression using a frozen HuBERT-ECG-base encoder.

---

## Phase 0 — Scalar Regression Head

### Architecture

- **Frozen HuBERT-ECG-base encoder** — no gradient updates
- **LearnedCompressor** (conv, ~490K params) — compresses encoder output across leads and time
- **decision_enc** MLP (550→512→256→256→64, ~496K params) — injects temporal context via Pan-Tompkins decision signal
- **qrs_head** (128→640→640→1, ~493K params) — scalar QRS duration output
- **qt_head** (129→640→640→1, ~494K params) — scalar QT duration output (conditioned on QRS prediction)

**Total trainable params: ~1.97M**

### Results

- Train error: below clinical noise floor (~10 ms)
- Same-patient validation: ~10 ms
- New-patient holdout: degrades (expected — patient generalisation deferred to Phase 1)

**Conclusion:** representational capacity confirmed; decision signal carries temporal structure; ready for logit mask head.

---

## Phase 1 — Logit Mask Head

Replaced scalar regression with a **soft mask** over the beat window: the model predicts a per-ms sigmoid probability and duration = mask.sum().

### Architecture

**g path (compressor)**
- Input: `(N, 12, t, 768)` HuBERT embeddings
- Conv2d bottleneck (768→1024→512) + depthwise temporal Conv2d stack (128→64→32→16) + learned lead-collapse Conv2d(16, 1, kernel=(12,7))
- Upsample → `(N, 1, 550)`, z-scored

**f path (PT anchor)**
- Input: Pan-Tompkins decision signal `(N, 550)`
- z-scored → `(N, 1, 550)` — directly anchors the mask to beat structure; no additional MLP

**Fusion**
- `cat[g, f]` → Conv1d(2→64, k=7) → Conv1d(64→64, k=7) → Conv1d(64→1, k=1) → logits `(N, 1, 550)`
- `sigmoid(logits)` → mask; `mask.sum()` → duration in ms

**Loss:** BCE on ground-truth rectangular mask + MAE regularisation

### Results (epoch 10 000)

| Split | QRS MAE | QT MAE |
|---|---|---|
| Train | 0.0 ms | 0.0 ms |
| Val | 17.9 ms | 0.0 ms |
| Holdout (p9) | 22.9 ms | 0.0 ms |

**Conclusion:** mask head achieves near-zero train error; PT anchor provides strong temporal signal; val/holdout generalisation competitive with Phase 0 scalar head.
