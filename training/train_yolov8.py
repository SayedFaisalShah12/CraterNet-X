import os
import torch
from ultralytics import YOLO
from pathlib import Path

def train_crater_detector(data_yaml, epochs=50, imgsz=640, batch=16):
    """
    Trains YOLOv8 on the Moon crater dataset.
    """
    # 1. Initialize YOLOv8 model
    # We use YOLOv8n (nano) for a balance of speed and performance
    model_type = 'yolov8n.pt'
    print(f"Loading pre-trained model: {model_type}")
    model = YOLO(model_type)

    # 2. Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 3. Setup Project Path
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / 'models' / 'stage1_detection'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training on {data_yaml}...")
    
    # 4. Run Training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(save_dir),
        name='moon_crater_yolov8',
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Model and logs saved to: {save_dir}/moon_crater_yolov8")
    
    # Return path to the best weight file
    best_model_path = save_dir / 'moon_crater_yolov8' / 'weights' / 'best.pt'
    return best_model_path

if __name__ == "__main__":
    # Path to data.yaml
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_YAML = str(PROJECT_ROOT / 'data' / 'moon' / 'craters' / 'data.yaml')
    
    # You can adjust epochs for testing (e.g., epochs=5)
    train_crater_detector(DATA_YAML, epochs=25, imgsz=640, batch=8)
