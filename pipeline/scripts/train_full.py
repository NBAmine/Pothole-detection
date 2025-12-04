import glob
import numpy as np
from ultralytics import YOLO

DATASET_ROOT = "pothole_dataset"

map_files = glob.glob("runs/pothole_fold_*_map.txt")
maps = [float(open(f).read().strip()) for f in map_files]

best_fold = np.argmax(maps) + 1
best_weights = f"runs/pothole_fold_{best_fold}/weights/best.pt"

# --- Create data.yaml -----
data_yaml=f"""
path: /kaggle/working/pothole_dataset
train: images/train
val: images/val
nc: 1
names: ['pothole']
"""

# --- Train model ---
model = YOLO(best_weights)
results = model.train(
    data=data_yaml,
    epochs=200,
    imgsz=640,
    batch=16,
    name="final_pothole",
    project="runs_final",
    patience=20,
    augment=True,
    exist_ok=True
)