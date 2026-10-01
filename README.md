deslopping my brain by writing a transformer from scratch in pytorch

how i did this: 
https://jalammar.github.io/illustrated-transformer/
https://nlp.seas.harvard.edu/annotated-transformer/#full-model
only use of AI was to understand transformer. no ai used to write code (exception of primitive autocomplete) (code has been formatted though)




results:

epoch 0: train loss 5.1533, val loss 4.0735
epoch 1: train loss 3.9400, val loss 3.6532
epoch 2: train loss 3.6145, val loss 3.4637
epoch 3: train loss 3.4126, val loss 3.3268
epoch 4: train loss 3.2652, val loss 3.2090
epoch 5: train loss 3.1532, val loss 3.1487
epoch 6: train loss 3.0606, val loss 3.0954
epoch 7: train loss 2.9847, val loss 3.0462
epoch 8: train loss 2.9182, val loss 3.0203
epoch 9: train loss 2.8582, val loss 2.9936
epoch 10: train loss 2.8061, val loss 2.9585
epoch 11: train loss 2.7595, val loss 2.9450
epoch 12: train loss 2.7173, val loss 2.9267
epoch 13: train loss 2.6768, val loss 2.9125
epoch 14: train loss 2.6413, val loss 2.9125
epoch 15: train loss 2.6074, val loss 2.8911
epoch 16: train loss 2.5739, val loss 2.8873
epoch 17: train loss 2.5446, val loss 2.8782
epoch 18: train loss 2.5159, val loss 2.8798
epoch 19: train loss 2.4922, val loss 2.8652

train BLEU (29000 sentences): 43.29 (good but but it's actually not as near perfect as ideal)
test BLEU (1000 sentences): 35.65
validation BLEU (1014 sentences): 36.18

near match test and validation is pretty cool because we selected checkpoint based on validation loss and validation isn't biased