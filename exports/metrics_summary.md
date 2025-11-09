# Metrics Summary

## baseline
- samples: 734
- accuracy: 0.9687
- macro-F1: 0.9689

Confusion matrix (rows=true, cols=pred):
```
[[238   0   0]
 [  8 243   4]
 [  0  11 230]]
```

## transformer_best
- samples: 734
- accuracy: 0.8093
- macro-F1: 0.8120

Confusion matrix (rows=true, cols=pred):
```
[[175  62   1]
 [ 25 211  19]
 [  5  28 208]]
```

## Logged Runs
- baseline/tfidf-logreg: macro-F1=0.8050088305616837 (TF-IDF + LogReg baseline)
- baseline-grid/tfidf-charwb_3_5: macro-F1=0.7895965126409101 (GridSearchCV 5-fold; best_params={'clf__C': 1.0, 'clf__class_weight': None})
- transformer/xlm-roberta-base: macro-F1=0.7902168405854463 (early_stop=2; class_weights=False)
- baseline/tfidf-logreg: macro-F1=0.8050088305616837 (TF-IDF + LogReg baseline)
- baseline-grid/tfidf-word_1_2: macro-F1=0.9689018288467438 (GridSearchCV 5-fold; best_params={'clf__C': 4.0, 'clf__class_weight': None})
- transformer/xlm-roberta-base: macro-F1=0.8120180430857343 (early_stop=2; class_weights=False)