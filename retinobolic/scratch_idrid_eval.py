import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '.')
from configs.config import CLASS_NAMES, CHECKPOINT_PATH, DEVICE, NUM_CLASSES
from src.model import build_model
from src.preprocess import get_val_transforms
from src.inference import _apply_calibrated_thresholds

class SimpleDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['file_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(row['mapped_grade'])

df = pd.read_csv('data/idrid/idrid_standardized.csv')
ds = SimpleDataset(df, transform=get_val_transforms())
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

model = build_model(architecture="efficientnet_b0", num_classes=NUM_CLASSES, pretrained=False, device=DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
model.eval()

all_probs, all_targets = [], []
with torch.no_grad():
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.numpy())

all_probs = np.vstack(all_probs)
all_targets = np.concatenate(all_targets)

preds = np.array([_apply_calibrated_thresholds(p) for p in all_probs])

acc = accuracy_score(all_targets, preds)
f1 = f1_score(all_targets, preds, average="macro")
rec = recall_score(all_targets, preds, average="macro")

print("=" * 65)
print("  IDRiD ZERO-SHOT EXTERNAL VALIDATION RESULTS")
print("=" * 65)
print(f"  Total External Test Images: {len(df)}")
print(f"  • External Accuracy     : {acc * 100:.2f}%")
print(f"  • Macro F1-Score        : {f1 * 100:.2f}%")
print(f"  • Macro Sensitivity     : {rec * 100:.2f}%")

print("\nPer-Class Sensitivity/Recall:")
for i, name in CLASS_NAMES.items():
    r = recall_score(all_targets, preds, labels=[i], average="macro", zero_division=0)
    print(f"  [{i}] {name:<15}: {r * 100:5.1f}%")
print("=" * 65)
