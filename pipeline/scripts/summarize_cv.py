import glob
import numpy as np

map_files = glob.glob("runs/pothole_fold_*_map.txt")
maps = [float(open(f).read().strip()) for f in map_files]

print("Cross-Validation Results (mAP@0.5:0.95):")
print(f"  Mean: {np.mean(maps):.4f}")
print(f"  Std:  {np.std(maps):.4f}")
print(f"  Best Fold: {np.argmax(maps) + 1} (mAP: {max(maps):.4f})")
