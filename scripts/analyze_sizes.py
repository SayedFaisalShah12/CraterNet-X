import os
import yaml
import numpy as np
from pathlib import Path

def analyze_sizes(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_yaml_path).parent
    train_images = dataset_root / data_config.get('train', 'train/images')
    train_labels = train_images.parent / 'labels'

    sizes = []
    
    label_files = list(train_labels.glob('*.txt'))
    print(f"Analyzing {len(label_files)} label files...")

    for lf_path in label_files:
        with open(lf_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # YOLO format: cls, x, y, w, h
                    w, h = float(parts[3]), float(parts[4])
                    # Assuming square-ish craters, we use max(w, h) or sqrt(w*h)
                    # Let's use max dimension as size proxy
                    size = max(w, h)
                    sizes.append(size)

    if not sizes:
        print("No labels found.")
        return

    sizes = np.array(sizes)
    print("\nSize distribution (normalized 0-1):")
    print(f"Min:    {sizes.min():.4f}")
    print(f"Max:    {sizes.max():.4f}")
    print(f"Mean:   {sizes.mean():.4f}")
    print(f"Median: {np.median(sizes):.4f}")
    print(f"25th percentile: {np.percentile(sizes, 25):.4f}")
    print(f"75th percentile: {np.percentile(sizes, 75):.4f}")
    
    # Suggested thresholds based on percentiles
    # Small: bottom 33%
    # Medium: middle 33%
    # Large: top 33%
    t1 = np.percentile(sizes, 33)
    t2 = np.percentile(sizes, 66)
    
    print(f"\nSuggested thresholds for classification:")
    print(f"Small:  size < {t1:.4f}")
    print(f"Medium: {t1:.4f} <= size < {t2:.4f}")
    print(f"Large:  size >= {t2:.4f}")

if __name__ == "__main__":
    DATA_YAML = r"e:\Artificial Intelligence\Z-Projects\CraterNet-X\data\moon\craters\data.yaml"
    analyze_sizes(DATA_YAML)
