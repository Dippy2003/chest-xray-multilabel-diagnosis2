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

## day 4 - per-class threshold tuning

flat 0.5 threshold was a bad fit for classes that are mostly negative (e.g. Pneumonia, ~1% positive).
swept thresholds 0.05-0.95 per class on the val set, picked whatever maximized that class's F1, then
applied those frozen thresholds to the test set (test set never used for picking thresholds).

macro F1, flat 0.5 vs tuned:

| model | macro AUC | f1 @ 0.5 | f1 tuned |
|---|---|---|---|
| baseline CNN (scratch) | 0.676 | 0.257 | 0.250 |
| resnet50 (transfer) | 0.755 | 0.308 | 0.318 |
| efficientnet-b0 (transfer) | 0.728 | 0.286 | 0.291 |

resnet50 tuned thresholds: No Finding 0.30, Atelectasis 0.65, Cardiomegaly 0.75, Effusion 0.50, Pneumonia 0.40

notes:
- AUC doesn't change (threshold-independent), only F1 moves, as expected
- resnet50 and efficientnet both improve with tuning - mainly No Finding and Atelectasis F1 go up since their optimal threshold isn't 0.5
- baseline's tuned F1 is actually slightly *worse* than flat 0.5 (0.250 vs 0.257) - its predictions are noisier (lowest AUC of the 3), so per-class thresholds picked on a small val set (841 images) overfit slightly instead of generalizing to test. transfer models have cleaner probability estimates so tuning helps them more
- resnet50 stays the best model after tuning too, now at 0.318 macro F1
