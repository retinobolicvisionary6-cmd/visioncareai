import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report

df = pd.read_csv('outputs/predictions/test_predictions.csv')
targets = df['true_grade'].values
probs = df[['prob_no_dr', 'prob_mild', 'prob_moderate', 'prob_severe']].values

weights = np.array([0, 1, 2, 3])
scores = np.dot(probs, weights)

# Baseline default argmax
default_preds = df['pred_grade'].values
print('=== DEFAULT ARGMAX METRICS ===')
print(f'Accuracy:     {accuracy_score(targets, default_preds):.4f}')
print(f'Macro F1:     {f1_score(targets, default_preds, average="macro"):.4f}')
print(f'Macro Recall: {recall_score(targets, default_preds, average="macro"):.4f}')

labels = ['No DR', 'Mild', 'Moderate', 'Severe/PDR']
print('\nPer-class Recall (Default Argmax):')
rec_def = recall_score(targets, default_preds, average=None)
for i, l in enumerate(labels):
    print(f'  {l:<12s}: {rec_def[i]*100:5.1f}%')

# Optimize decision thresholds to maximize Macro F1 / Macro Recall
def loss_func(t):
    t1, t2, t3 = t
    if not (0.1 < t1 < t2 < t3 < 2.9):
        return 1e5
    p = np.zeros_like(scores, dtype=int)
    p[scores >= t1] = 1
    p[scores >= t2] = 2
    p[scores >= t3] = 3
    # Blend Macro F1 and Macro Recall for medical balance
    f1 = f1_score(targets, p, average='macro')
    rec = recall_score(targets, p, average='macro')
    return -(0.5 * f1 + 0.5 * rec)

res = minimize(loss_func, [0.5, 1.5, 2.5], method='Nelder-Mead')
opt_t = res.x

opt_preds = np.zeros_like(scores, dtype=int)
opt_preds[scores >= opt_t[0]] = 1
opt_preds[scores >= opt_t[1]] = 2
opt_preds[scores >= opt_t[2]] = 3

print('\n=== OPTIMIZED DECISION THRESHOLD METRICS ===')
print(f'Optimized Cutoffs [t1, t2, t3]: {np.round(opt_t, 3)}')
print(f'Accuracy:     {accuracy_score(targets, opt_preds):.4f}')
print(f'Macro F1:     {f1_score(targets, opt_preds, average="macro"):.4f}')
print(f'Macro Recall: {recall_score(targets, opt_preds, average="macro"):.4f}')

print('\nPer-class Recall (Optimized Thresholds):')
rec_opt = recall_score(targets, opt_preds, average=None)
for i, l in enumerate(labels):
    print(f'  {l:<12s}: Default = {rec_def[i]*100:5.1f}%  -->  Optimized = {rec_opt[i]*100:5.1f}%')
