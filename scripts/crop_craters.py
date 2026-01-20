import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm

def crop_craters(data_yaml_path, thresholds=(0.05, 0.10), padding=0.1):
    """
    Crops craters from images based on YOLO labels and categorizes them by size.
    """
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_yaml_path).parent
    project_root = dataset_root.parent.parent.parent
    output_base = project_root / 'data' / 'moon' / 'classification'
    
    t1, t2 = thresholds
    
    splits = {
        'train': dataset_root / data_config.get('train', 'train/images'),
        'val': dataset_root / data_config.get('val', 'valid/images'),
        'test': dataset_root / data_config.get('test', 'test/images')
    }

    print(f"Starting cropping process. Output: {output_base}")

    for split_name, img_dir in splits.items():
        if not img_dir.exists():
            continue
        
        label_dir = img_dir.parent / 'labels'
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        print(f"\nProcessing {split_name} split ({len(image_files)} images)...")
        
        for img_path in image_files:
            label_path = label_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h_img, w_img = img.shape[:2]
            
            with open(label_path, 'r') as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    _, x_norm, y_norm, w_norm, h_norm = map(float, parts)
                    
                    # Determine size category
                    size = max(w_norm, h_norm)
                    if size < t1:
                        category = 'small'
                    elif size < t2:
                        category = 'medium'
                    else:
                        category = 'large'
                    
                    # Convert normalized to pixel coordinates
                    x_center = x_norm * w_img
                    y_center = y_norm * h_img
                    width = w_norm * w_img
                    height = h_norm * h_img
                    
                    # Add padding
                    width_p = width * (1 + padding)
                    height_p = height * (1 + padding)
                    
                    x1 = int(max(0, x_center - width_p / 2))
                    y1 = int(max(0, y_center - height_p / 2))
                    x2 = int(min(w_img, x_center + width_p / 2))
                    y2 = int(min(h_img, y_center + height_p / 2))
                    
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    # Save crop
                    save_dir = output_base / split_name / category
                    os.makedirs(save_dir, exist_ok=True)
                    
                    save_name = f"{img_path.stem}_crater_{idx}.jpg"
                    cv2.imwrite(str(save_dir / save_name), crop)

    print("\nCropping complete!")
    
    # Print statistics
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} stats:")
        for cat in ['small', 'medium', 'large']:
            count = len(list((output_base / split / cat).glob('*.jpg'))) if (output_base / split / cat).exists() else 0
            print(f"  - {cat}: {count}")

if __name__ == "__main__":
    DATA_YAML = r"e:\Artificial Intelligence\Z-Projects\CraterNet-X\data\moon\craters\data.yaml"
    # Using thresholds from analysis: 0.05 and 0.10
    crop_craters(DATA_YAML, thresholds=(0.0516, 0.1022))
