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
