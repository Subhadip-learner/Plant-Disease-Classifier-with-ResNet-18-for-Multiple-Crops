# train_plant_disease.py
import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# ----- CONFIG -----
DATASET_ID = "Saon110/bd-crop-vegetable-plant-disease-dataset"
OUTPUT_DIR = "trained_model"
BATCH_SIZE = 32
NUM_EPOCHS = 8
LR = 3e-4
IMG_SIZE = 224

# On Windows, if you still have issues set NUM_WORKERS = 0
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- PyTorch Dataset wrapper for HF dataset -----
class HFDataset(Dataset):
    def __init__(self, hf_dataset, transforms=None):
        self.ds = hf_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        # common column name: 'image' -> PIL.Image, or path.
        img = item.get('image') or item.get('img') or item.get('image_path')
        # If it's a string path, open it; if it's a PIL image (common), ensure RGB
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        else:
            # datasets library sometimes stores a dict with "bytes" or an Image object
            if isinstance(img, Image.Image):
                img = img.convert('RGB')
            else:
                # try convert from ndarray-like
                img = Image.fromarray(np.array(img)).convert('RGB')
        label = item['label']
        if self.transforms:
            img = self.transforms(img)
        return img, int(label)

# ----- transforms -----
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ----- training / evaluation utilities -----
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="val", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

# ----- main workflow (protected for Windows multiprocessing) -----
def main():
    # Improve cuDNN performance where applicable
    torch.backends.cudnn.benchmark = True

    print("Loading dataset:", DATASET_ID)
    ds = load_dataset(DATASET_ID)  # will download if not present

    print("Dataset splits found:", ds.keys())

    # Use existing valid/test splits if present; otherwise split train -> train/valid
    if 'train' in ds and 'valid' not in ds and 'validation' not in ds:
        # many HF datasets use 'train' only; create a validation split
        ds = ds['train'].train_test_split(test_size=0.1, seed=42)
        train_ds = ds['train']
        val_ds = ds['test']
    else:
        train_ds = ds['train']
        # prefer 'valid' or 'validation' or fallback to 'test'
        if 'valid' in ds:
            val_ds = ds['valid']
        elif 'validation' in ds:
            val_ds = ds['validation']
        else:
            val_ds = ds['test']

    # determine class names
    features = train_ds.features
    if 'label' in features and hasattr(features['label'], 'names'):
        CLASS_NAMES = features['label'].names
    else:
        # fallback: gather distinct labels (may be slower)
        labels_list = list(set(train_ds['label']))
        CLASS_NAMES = [str(x) for x in sorted(labels_list)]
    NUM_CLASSES = len(CLASS_NAMES)
    print("Detected classes:", NUM_CLASSES)

    # build datasets and dataloaders
    train_dataset = HFDataset(train_ds, transforms=train_tf)
    val_dataset = HFDataset(val_ds, transforms=val_tf)

    # On Windows, if you experience multiprocessing errors, set NUM_WORKERS = 0 above
    print(f"Creating DataLoaders with batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # build model (transfer learning from ImageNet)
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.2)

    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} - LR: {scheduler.get_last_lr()}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} || Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_names': CLASS_NAMES,
                'epoch': epoch
            }, os.path.join(OUTPUT_DIR, 'best_resnet18.pth'))
            print(f"Saved best model (val_acc={best_val_acc:.4f}) -> {os.path.join(OUTPUT_DIR, 'best_resnet18.pth')}")

        scheduler.step()

    # save final checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': CLASS_NAMES,
        'epoch': NUM_EPOCHS
    }, os.path.join(OUTPUT_DIR, 'final_resnet18.pth'))
    print("\nTraining finished. Best val acc:", best_val_acc)
    print("Checkpoints saved in:", OUTPUT_DIR)

# ------------------- run -------------------
if __name__ == "__main__":
    # Needed for Windows to safely spawn child processes
    multiprocessing.freeze_support()
    main()
