import sys
import torch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

sys.path.insert(0, '.')
from src.model import build_model
from src.dataset import FundusDataset
from src.preprocess import get_val_transforms
from torch.utils.data import DataLoader
from configs.config import SPLITS_DIR, CHECKPOINT_PATH, DEVICE, RAW_DATA_DIR, NUM_CLASSES

# Load model
model = build_model(architecture='efficientnet_b0', num_classes=NUM_CLASSES, pretrained=False, device=DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Load validation data
val_df = pd.read_csv(SPLITS_DIR / 'val.csv')
ds_val = FundusDataset(val_df, img_dir=RAW_DATA_DIR / 'train_images', transform=get_val_transforms())
loader_val = DataLoader(ds_val, batch_size=32, shuffle=False)

val_probs, val_targets = [], []
with torch.no_grad():
    for imgs, targets in loader_val:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        val_probs.append(probs)
        val_targets.append(targets.numpy())

val_probs = np.vstack(val_probs)
val_targets = np.concatenate(val_targets)

# Default argmax baseline on validation
val_preds_default = np.argmax(val_probs, axis=1)
print('=== DEFAULT ARGMAX (VALIDATION SET) ===')
print(f'Accuracy:     {accuracy_score(val_targets, val_preds_default):.4f}')
print(f'Macro F1:     {f1_score(val_targets, val_preds_default, average="macro"):.4f}')
print(f'Macro Recall: {recall_score(val_targets, val_preds_default, average="macro"):.4f}')

# Method: Expected DR Value + Cutoff Threshold Optimization
weights = np.array([0, 1, 2, 3])
val_scores = np.dot(val_probs, weights)

def loss_func(thresholds):
    t1, t2, t3 = thresholds
    if not (0.1 < t1 < t2 < t3 < 2.9):
        return 1e5
    preds = np.zeros_like(val_scores, dtype=int)
    preds[val_scores >= t1] = 1
    preds[val_scores >= t2] = 2
    preds[val_scores >= t3] = 3
    return -f1_score(val_targets, preds, average='macro')

res = minimize(loss_func, [0.5, 1.5, 2.5], method='Nelder-Mead')
opt_t = res.x
print('\n=== OPTIMIZED CUTOFF THRESHOLDS (FOUND ON VALIDATION SET) ===')
print(f'Optimized Cutoffs [t1, t2, t3]: {np.round(opt_t, 4)}')

val_preds_opt = np.zeros_like(val_scores, dtype=int)
val_preds_opt[val_scores >= opt_t[0]] = 1
val_preds_opt[val_scores >= opt_t[1]] = 2
val_preds_opt[val_scores >= opt_t[2]] = 3

print(f'Optimized Val Accuracy:     {accuracy_score(val_targets, val_preds_opt):.4f}')
print(f'Optimized Val Macro F1:     {f1_score(val_targets, val_preds_opt, average="macro"):.4f}')
print(f'Optimized Val Macro Recall: {recall_score(val_targets, val_preds_opt, average="macro"):.4f}')

# Evaluate on HELD-OUT TEST SET (No Data Leakage)
test_df = pd.read_csv(SPLITS_DIR / 'test.csv')
ds_test = FundusDataset(test_df, img_dir=RAW_DATA_DIR / 'train_images', transform=get_val_transforms())
loader_test = DataLoader(ds_test, batch_size=32, shuffle=False)

test_probs, test_targets = [], []
with torch.no_grad():
    for imgs, targets in loader_test:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        test_probs.append(probs)
        test_targets.append(targets.numpy())

test_probs = np.vstack(test_probs)
test_targets = np.concatenate(test_targets)

test_preds_default = np.argmax(test_probs, axis=1)

test_scores = np.dot(test_probs, weights)
test_preds_opt = np.zeros_like(test_scores, dtype=int)
test_preds_opt[test_scores >= opt_t[0]] = 1
test_preds_opt[test_scores >= opt_t[1]] = 2
test_preds_opt[test_scores >= opt_t[2]] = 3

print('\n=== HELD-OUT TEST SET PERFORMANCE COMPARISON ===')
print(f'Default Argmax  --> Accuracy: {accuracy_score(test_targets, test_preds_default):.4f} | Macro F1: {f1_score(test_targets, test_preds_default, average="macro"):.4f} | Macro Recall: {recall_score(test_targets, test_preds_default, average="macro"):.4f}')
print(f'Optimized Rule  --> Accuracy: {accuracy_score(test_targets, test_preds_opt):.4f} | Macro F1: {f1_score(test_targets, test_preds_opt, average="macro"):.4f} | Macro Recall: {recall_score(test_targets, test_preds_opt, average="macro"):.4f}')

print('\n=== PER-CLASS RECALL COMPARISON (TEST SET) ===')
labels = ['No DR', 'Mild', 'Moderate', 'Severe/PDR']
rec_def = recall_score(test_targets, test_preds_default, average=None)
rec_opt = recall_score(test_targets, test_preds_opt, average=None)
for i, l in enumerate(labels):
    print(f'  {l:<12s}: Default Recall = {rec_def[i]*100:5.1f}%  -->  Optimized Recall = {rec_opt[i]*100:5.1f}%')
