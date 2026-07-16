# LungLens — Baseline vs Fine-tuned

| Metric | Baseline (frozen) | Fine-tuned (all layers) |
|---|---|---|
| Test accuracy | 85.10% | 93.84% |
| Best val accuracy | 84.76% | 94.33% |
| Trainable params | 2,052 | 11,178,564 |
| Epochs run | 11 | 12 |
| Train time (min) | 11.4 | 21.3 |
| Augmentation | False | True |
| Scheduler | None | CosineAnnealingLR |

**Absolute accuracy gap: +8.74 percentage points** | Relative improvement: +10.3%
