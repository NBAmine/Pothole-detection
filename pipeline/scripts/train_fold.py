import os
import yaml
import pandas as pd
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True)
args = parser.parse_args()

FOLD_NUM = args.fold

DATASET_DIR = os.path.abspath("pothole_dataset")

yaml_path = f"{DATASET_DIR}/folds/fold_{FOLD_NUM}_data.yaml"

# --- Train ---
model = YOLO("yolov8n.pt")
results = model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    name=f"pothole_fold_{FOLD_NUM}",
    project="runs",
    patience=20,
    augment=True,
    exist_ok=True,
    verbose=False,
    plots=False,
)

# Save mAP
map_score = results.results_dict['metrics/mAP50-95(B)']
with open(f"runs/pothole_fold_{FOLD_NUM}_map.txt", "w") as f:
    f.write(str(map_score))

print(f"Fold {FOLD_NUM} mAP@0.5:0.95 = {map_score:.4f}")