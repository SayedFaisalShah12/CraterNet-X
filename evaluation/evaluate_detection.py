import os
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

def evaluate_and_visualize(model_path, data_yaml, split='test', num_samples=5):
    """
    Evaluates the YOLOv8 model and visualizes results on sample images.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 1. Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # 2. Run validation/evaluation
    print(f"Running evaluation on '{split}' split...")
    results = model.val(data=data_yaml, split=split)
    
    # 3. Print key metrics
    print("\n" + "="*50)
    print("DETECTION METRICS")
    print("="*50)
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print("="*50)

    # 4. Visualize sample predictions
    print(f"\nVisualizing {num_samples} sample predictions...")
    
    # Path to images in the specified split
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    img_dir = Path(data_yaml).parent / data_config.get(split, f'{split}/images')
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    
    if not image_files:
        print(f"No images found in {img_dir}")
        return

    # Select samples
    samples = image_files[:min(num_samples, len(image_files))]
    
    # Create output directory for visualizations
    vis_dir = Path(__file__).resolve().parent.parent / 'evaluation' / 'results'
    os.makedirs(vis_dir, exist_ok=True)

    for i, img_path in enumerate(samples):
        # Run inference
        pred_results = model.predict(str(img_path))
        
        # Plot and save
        res_plotted = pred_results[0].plot()
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(res_plotted_rgb)
        plt.title(f"Detection Result: {img_path.name}")
        plt.axis('off')
        
        save_path = vis_dir / f"detection_{img_path.name}"
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved visualization: {save_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Defaults (assuming training finished)
    MODEL_PATH = PROJECT_ROOT / 'models' / 'stage1_detection' / 'moon_crater_yolov8' / 'weights' / 'best.pt'
    DATA_YAML = PROJECT_ROOT / 'data' / 'moon' / 'craters' / 'data.yaml'
    
    # Check if model exists before running
    if MODEL_PATH.exists():
        evaluate_and_visualize(str(MODEL_PATH), str(DATA_YAML))
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Run training first.")
