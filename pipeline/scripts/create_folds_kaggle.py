from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

IMGS_DIR = "/kaggle/input/pothole-dataset/images/train"
LABELS_DIR = "/kaggle/input/pothole-dataset/images/train"

K = 5  # folds

# Count potholes per image
def count_boxes(ann_path):
    if not os.path.exists(ann_path):
        return 0
    with open(ann_path) as f:
        return len([line for line in f.read().strip().split('\n') if line.strip()])

# Build DataFrame
imgs = [f for f in os.listdir(IMGS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
data = []
for img in tqdm(imgs, desc="Counting boxes"):
    ann = os.path.join(LABELS_DIR, os.path.splitext(img)[0] + '.txt')
    num_boxes = count_boxes(ann)
    data.append({'filename': img, 'num_boxes': num_boxes})

df = pd.DataFrame(data)

# Stratify: group negatives + bins for positives
bins = [1, 3, 6, 999]
df['stratify'] = np.where(df['num_boxes'] == 0, 0, np.digitize(df['num_boxes'], bins))

# 5-Fold CV
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

OUT_ROOT = "/kaggle/working/pothole_dataset/folds"
os.makedirs(OUT_ROOT, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify'])):
    
    fold_num = fold + 1
    
    train_files = df.iloc[train_idx]['filename'].tolist()
    val_files   = df.iloc[val_idx]['filename'].tolist()

    # --- Write .txt path lists ---
    train_txt = os.path.join(OUT_ROOT, f"fold_{fold_num}_train.txt")
    val_txt   = os.path.join(OUT_ROOT, f"fold_{fold_num}_val.txt")

    with open(train_txt, 'w') as f:
        for file in train_files:
            f.write(os.path.join(IMGS_DIR, file) + '\n')
    with open(val_txt, 'w') as f:
        for file in val_files:
            f.write(os.path.join(IMGS_DIR, file) + '\n')
            
    # --- Write data.yaml ---
    yaml_content = f"""
path: /kaggle/working/pothole_dataset
train: folds/fold_{fold_num}_train.txt
val:   folds/fold_{fold_num}_val.txt
nc: 1
names: ['pothole']
    """
    
    yaml_path = os.path.join(OUT_ROOT, f"fold_{fold_num}_data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())

print(f"Created {K}-fold splits. Each fold: ~{len(df)//K} val images.")