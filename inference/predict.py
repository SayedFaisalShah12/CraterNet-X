import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

class CraterNetX:
    """
    Complete Two-Stage Crater Detection and Classification Pipeline.
    Stage 1: YOLOv8 Detection
    Stage 2: ResNet-50 Classification
    """
    def __init__(self, detection_model_path, classification_model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Initializing CraterNet-X on {self.device}...")
        
        # Load Detection Model
        self.detector = YOLO(detection_model_path)
        
        # Load Classification Model
        self.classifier = models.resnet50(weights=None)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Linear(num_ftrs, 3) # small, medium, large
        
        self.classifier.load_state_dict(torch.load(classification_model_path, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.id_to_class = {0: 'large', 1: 'medium', 2: 'small'} # Verify with training labels
        # Note: datasets.ImageFolder alphabetical order usually: large, medium, small
        # Adjust based on your actual data/moon/classification directory structure
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, conf=0.25):
        """
        Runs the full pipeline on a single image.
        """
        img0 = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        
        # Stage 1: Detection
        det_results = self.detector.predict(image_path, conf=conf, verbose=False)[0]
        
        detections = []
        for box in det_results.boxes:
            # Get box coordinates (x1, y1, x2, y2)
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            # Crop detected crater
            # Add small padding
            h_orig, w_orig = img0.shape[:2]
            pad = 5
            px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
            px2, py2 = min(w_orig, x2 + pad), min(h_orig, y2 + pad)
            
            crop = img_rgb[py1:py2, px1:px2]
            if crop.size == 0:
                continue
                
            # Stage 2: Classification
            crop_pil = Image.fromarray(crop)
            input_tensor = self.preprocess(crop_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.classifier(input_tensor)
                _, pred = torch.max(output, 1)
                size_class = self.id_to_class[pred.item()]
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'size_class': size_class,
                'confidence': float(box.conf[0])
            })
            
        return detections, img0

    def visualize(self, image_path, detections, output_path=None):
        """
        Visualizes detections with size classification on the image.
        """
        img = cv2.imread(image_path)
        colors = {
            'small': (0, 255, 0),    # Green
            'medium': (255, 255, 0), # Yellow
            'large': (0, 0, 255)     # Red
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls = det['size_class']
            conf = det['confidence']
            color = colors.get(cls, (255, 255, 255))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} ({conf:.2f})"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Visualization saved to {output_path}")
        
        return img

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Paths to best models
    DET_MODEL = PROJECT_ROOT / 'models' / 'stage1_detection' / 'moon_crater_yolov8' / 'weights' / 'best.pt'
    CLS_MODEL = PROJECT_ROOT / 'models' / 'stage2_classification' / 'crater_classifier_resnet50.pt'
    
    if DET_MODEL.exists() and CLS_MODEL.exists():
        pipeline = CraterNetX(str(DET_MODEL), str(CLS_MODEL))
        
        # Test on a random image from test set
        test_img_dir = PROJECT_ROOT / 'data' / 'moon' / 'craters' / 'test' / 'images'
        test_images = list(test_img_dir.glob('*.jpg'))
        
        if test_images:
            img_path = str(test_images[0])
            print(f"Running inference on {img_path}...")
            results, _ = pipeline.predict(img_path)
            
            # Save visualization
            os.makedirs(PROJECT_ROOT / 'inference' / 'results', exist_ok=True)
            out_path = PROJECT_ROOT / 'inference' / 'results' / 'prediction_test.jpg'
            pipeline.visualize(img_path, results, str(out_path))
    else:
        print("Models not found. Training might still be in progress.")
