import os
import cv2
from ultralytics import YOLO
from pathlib import Path

def test_inference():
    project_root = Path(r'e:\Artificial Intelligence\Z-Projects\CraterNet-X')
    model_path = project_root / 'models' / 'stage1_detection' / 'moon_crater_yolov8' / 'weights' / 'best.pt'
    
    model = YOLO(str(model_path))
    temp_dir = project_root / 'app' / 'temp'
    images = ['1.jpg', '2.jpg', '3.jpg']
    
    for img_name in images:
        img_path = temp_dir / img_name
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        print(f"Image: {img_name}, Resolution: {img.shape}")
        results = model.predict(str(img_path), conf=0.01, verbose=False)[0]
        if len(results.boxes) > 0:
            best_conf = float(results.boxes.conf.max())
            print(f"  Best Confidence: {best_conf:.4f}")
        else:
            print("  No detections even at 0.01")

if __name__ == "__main__":
    test_inference()
