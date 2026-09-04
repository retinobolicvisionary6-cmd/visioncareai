import json

with open('outputs/metrics/evaluation_report.json') as f:
    r = json.load(f)

print('=== PER-CLASS METRICS ===')
for cls, m in r['per_class_metrics'].items():
    print(f'\n  {cls}:')
    for k, v in m.items():
        print(f'    {k:25s}: {v}')

print('\n=== CONFUSION MATRIX ===')
cm = r['confusion_matrix']
labels = ['No DR', 'Mild', 'Moderate', 'Severe/PDR']
print(f'                  Predicted:')
header = '  '.join(f'{l:>10s}' for l in labels)
print(f'  Actual \\ Pred  | {header}')
print(f'  {"="*65}')
for i, row in enumerate(cm):
    vals = '  '.join(f'{v:>10d}' for v in row)
    print(f'  {labels[i]:>14s} | {vals}')
