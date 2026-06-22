# model comparison - day 3

trained on the 5606 image sample dataset, 70/15/15 split, same pos_weight setup for all 3.

| model | macro AUC | macro F1 | epochs run | trainable params |
|---|---|---|---|---|
| baseline CNN (scratch) | 0.677 | 0.287 | 30 (no early stop) | ~1.0M |
| resnet50 (transfer) | 0.755 | 0.308 | 13 (early stopped) | ~15.0M |
| efficientnet-b0 (transfer) | 0.728 | 0.286 | 16 (early stopped) | ~1.1M |

per-class AUC:

| class | baseline | resnet50 | efficientnet-b0 |
|---|---|---|---|
| No Finding | 0.656 | 0.706 | 0.703 |
| Atelectasis | 0.729 | 0.744 | 0.744 |
| Cardiomegaly | 0.522 | 0.710 | 0.736 |
| Effusion | 0.733 | 0.795 | 0.790 |
| Pneumonia | 0.745 | 0.819 | 0.668 |

notes:
- both transfer models beat the scratch baseline, biggest gain on Cardiomegaly (smallest class, ~140 positives) - pretrained features help most where we have least data
- resnet50 wins overall but efficientnet-b0 gets close with 13x fewer trainable params, and is more competitive everywhere except pneumonia
- resnet50 stopped early at epoch 13 (best epoch 8), efficientnet at epoch 16 (best epoch 11) - resnet overfits faster since it has way more trainable params (layer4 unfrozen is bigger than efficientnet's unfrozen tail)
- f1 still low everywhere - flat 0.5 threshold problem, not a model problem. day 4 fixes this with per-class threshold tuning
