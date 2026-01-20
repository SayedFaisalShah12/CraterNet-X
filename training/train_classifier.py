import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import time
import copy

def train_size_classifier(data_dir, num_epochs=25, batch_size=32, learning_rate=0.001):
    """
    Trains a ResNet-50 classifier to categorize crater sizes.
    """
    # 1. Data Transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Load Datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Classes found: {class_names}")
    print(f"Training on {dataset_sizes['train']} images, Validating on {dataset_sizes['val']} images.")

    # 3. Setup Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize ResNet-50
    # Option: weights=models.ResNet50_Weights.DEFAULT (ImageNet)
    # We use ImageNet weights for better feature extraction on small datasets
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Modify final layer for 3 classes: small, medium, large
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 5. Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\nStarting Stage-2 Training (Classification)...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Training complete. Best val Acc: {best_acc:4f}')

    # Save the model
    save_dir = Path(__file__).resolve().parent.parent / 'models' / 'stage2_classification'
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / 'crater_classifier_resnet50.pt'
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to: {save_path}")
    
    return save_path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = str(PROJECT_ROOT / 'data' / 'moon' / 'classification')
    
    if os.path.exists(DATA_DIR):
        train_size_classifier(DATA_DIR, num_epochs=15, batch_size=32)
    else:
        print(f"Error: Data directory not found at {DATA_DIR}. Run crop_craters.py first.")
