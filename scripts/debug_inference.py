import os
import cv2
from ultralytics import YOLO
from pathlib import Path

def test_inference():
    project_root = Path(r'e:\Artificial Intelligence\Z-Projects\CraterNet-X')
    model_path = project_root / 'models' / 'stage1_detection' / 'moon_crater_yolov8' / 'weights' / 'best.pt'
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    model = YOLO(str(model_path))
    temp_dir = project_root / 'app' / 'temp'
    images = ['1.jpg', '2.jpg', '3.jpg']
    
    thresholds = [0.1, 0.05, 0.01]
    
    for img_name in images:
        img_path = temp_dir / img_name
        if not img_path.exists():
            print(f"Image {img_name} not found.")
            continue
            
        print(f"\n--- Testing {img_name} ---")
        img = cv2.imread(str(img_path))
        print(f"Resolution: {img.shape}")
        
        for t in thresholds:
            results = model.predict(str(img_path), conf=t, verbose=False)[0]
            print(f"Threshold {t}: Found {len(results.boxes)} detections")
            if len(results.boxes) > 0:
                for box in results.boxes:
                    print(f"  - Conf: {float(box.conf[0]):.4f}, Class: {int(box.cls[0])}")

if __name__ == "__main__":
    test_inference()
