import os
import yaml
import cv2
from pathlib import Path

def verify_dataset(data_yaml_path):
    """
    Verifies the integrity and format of the YOLO dataset.
    """
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_yaml_path).parent
    splits = {
        'train': dataset_root / data_config.get('train', 'train/images'),
        'val': dataset_root / data_config.get('val', 'valid/images'),
        'test': dataset_root / data_config.get('test', 'test/images')
    }

    print(f"Starting verification for dataset at: {dataset_root}")
    
    overall_stats = {
        'total_images': 0,
        'total_labels': 0,
        'missing_labels': [],
        'corrupt_images': [],
        'invalid_labels': [],
        'class_distribution': {}
    }

    for split_name, image_dir in splits.items():
        if not image_dir.exists():
            print(f"Warning: Split '{split_name}' directory not found: {image_dir}")
            continue

        label_dir = image_dir.parent / 'labels'
        print(f"\nChecking split: {split_name} (Images: {image_dir}, Labels: {label_dir})")

        image_files = list(image_dir.glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        split_stats = {
            'images': len(image_files),
            'labels': 0,
            'missing': 0,
            'invalid': 0
        }

        print(f"Verifying {len(image_files)} images in {split_name}...")
        for i, img_path in enumerate(image_files):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(image_files)} images...")
            overall_stats['total_images'] += 1
            
            # 1. Check if image is corrupt
            img = cv2.imread(str(img_path))
            if img is None:
                overall_stats['corrupt_images'].append(str(img_path))
                continue

            # 2. Check for corresponding label
            label_path = label_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                overall_stats['missing_labels'].append(str(img_path))
                split_stats['missing'] += 1
                continue

            split_stats['labels'] += 1
            overall_stats['total_labels'] += 1

            # 3. Validate YOLO format
            try:
                with open(label_path, 'r') as lf:
                    lines = lf.readlines()
                    for line_idx, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            overall_stats['invalid_labels'].append(f"{label_path} (Line {line_idx+1}: Expected 5 values)")
                            split_stats['invalid'] += 1
                            continue
                        
                        cls_id, x, y, w, h = map(float, parts)
                        
                        # Check class ID (Moon dataset should only have class 0)
                        cls_id = int(cls_id)
                        overall_stats['class_distribution'][cls_id] = overall_stats['class_distribution'].get(cls_id, 0) + 1
                        
                        # Check normalization
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            overall_stats['invalid_labels'].append(f"{label_path} (Line {line_idx+1}: Coordinates not normalized)")
                            split_stats['invalid'] += 1
            except Exception as e:
                overall_stats['invalid_labels'].append(f"{label_path} (Error: {str(e)})")
                split_stats['invalid'] += 1

        print(f"[OK] {split_name} Summary: {split_stats['images']} images, {split_stats['labels']} labels, {split_stats['missing']} missing, {split_stats['invalid']} invalid.")

    # Final Report
    print("\n" + "="*50)
    print("DATASET VERIFICATION REPORT")
    print("="*50)
    print(f"Total Images Checked:  {overall_stats['total_images']}")
    print(f"Total Labels Found:    {overall_stats['total_labels']}")
    print(f"Corrupt Images:        {len(overall_stats['corrupt_images'])}")
    print(f"Missing Labels:        {len(overall_stats['missing_labels'])}")
    print(f"Invalid Label Format:  {len(overall_stats['invalid_labels'])}")
    
    print("\nClass Distribution:")
    for cls_id, count in overall_stats['class_distribution'].items():
        name = data_config['names'].get(cls_id, f"Unknown({cls_id})")
        print(f"  - Class {cls_id} ({name}): {count} occurrences")

    if overall_stats['missing_labels'] or overall_stats['invalid_labels'] or overall_stats['corrupt_images']:
        print("\nVERIFICATION FAILED: Issues found. Please check logs.")
        if overall_stats['missing_labels']:
            print(f"First 5 missing labels: {overall_stats['missing_labels'][:5]}")
        if overall_stats['invalid_labels']:
            print(f"First 5 invalid labels: {overall_stats['invalid_labels'][:5]}")
    else:
        print("\nVerification Passed: Dataset is clean and ready for Stage-1 training!")
    print("="*50)

if __name__ == "__main__":
    # Use relative path if possible or full path
    DATA_YAML = r"e:\Artificial Intelligence\Z-Projects\CraterNet-X\data\moon\craters\data.yaml"
    verify_dataset(DATA_YAML)
