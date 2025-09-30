
---

# 🌿 Plant Disease Classifier with ResNet-18 📊

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)  
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23000000?style=for-the-badge&logo=huggingface&logoColor=white)  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)  


> A robust, transfer-learning-based deep learning pipeline for classifying plant diseases using the **BD-Crop Vegetable Plant Disease Dataset**. Built with PyTorch and Hugging Face Datasets.

---

## 📸 Demo / Visuals


### Supported Crops and Diseases

This model can classify **85 different classes** across **14 different crops**. Here's the complete list of what the system can detect:

---

### 🍌 **Banana** (9 classes)
- **Banana_Black_Pitting_or_Banana_Rust**
- **Banana_Crown_Rot**
- **Banana_Healthy**
- **Banana_fungal_disease**
- **Banana_leaf_Banana_Scab_Moth**
- **Banana_leaf_Black_Sigatoka**
- **Banana_leaf_Healthy**
- **Banana_leaf__Black_Leaf_Streak**
- **Banana_leaf__Panama_Disease**

---

### 🥦 **Cauliflower** (4 classes)
- **Cauliflower_Bacterial_spot_rot**
- **Cauliflower_Black_Rot**
- **Cauliflower_Downy_Mildew**
- **Cauliflower_Healthy**

---

### 🌽 **Corn** (4 classes)
- **Corn_Blight**
- **Corn_Common_Rust**
- **Corn_Gray_Leaf_Spot**
- **Corn_Healthy**

---

### 🧵 **Cotton** (4 classes)
- **Cotton_Aphids**
- **Cotton_Army_worm**
- **Cotton_Bacterial_blight**
- **Cotton_Healthy**

---

### 🍈 **Guava** (10 classes)
- **Guava_fruit_Anthracnose**
- **Guava_fruit_Healthy**
- **Guava_fruit_Scab**
- **Guava_fruit_Styler_end_root**
- **Guava_leaf_Anthracnose**
- **Guava_leaf_Canker**
- **Guava_leaf_Dot**
- **Guava_leaf_Healthy**
- **Guava_leaf_Rust**

---

### 🌿 **Jute** (3 classes)
- **Jute_Cescospora_Leaf_Spot**
- **Jute_Golden_Mosaic**
- **Jute_Healthy_Leaf**

---

### 🥭 **Mango** (7 classes)
- **Mango_Anthracnose**
- **Mango_Bacterial_Canker**
- **Mango_Cutting_Weevil**
- **Mango_Gall_Midge**
- **Mango_Healthy**
- **Mango_Powdery_Mildew**
- **Mango_Sooty_Mould**
- **Mango_die_back**

---

### 🍈 **Papaya** (8 classes)
- **Papaya_Anthracnose**
- **Papaya_BacterialSpot**
- **Papaya_Curl**
- **Papaya_Healthy**
- **Papaya_Mealybug**
- **Papaya_Mite_disease**
- **Papaya_Mosaic**
- **Papaya_Ringspot**

---

### 🥔 **Potato** (10 classes)
- **Potato_Black_Scurf**
- **Potato_Blackleg**
- **Potato_Blackspot_Bruising**
- **Potato_Brown_Rot**
- **Potato_Common_Scab**
- **Potato_Dry_Rot**
- **Potato_Healthy_Potatoes**
- **Potato_Miscellaneous**
- **Potato_Pink_Rot**
- **Potato_Soft_Rot**

---

### 🍚 **Rice** (9 classes)
- **Rice_Blast**
- **Rice_Brownspot**
- **Rice_Tungro**
- **Rice_bacterial_leaf_blight**
- **Rice_bacterial_leaf_streak**
- **Rice_bacterial_panicle_blight**
- **Rice_dead_heart**
- **Rice_downy_mildew**
- **Rice_hispa**
- **Rice_normal**

---

### 🎋 **Sugarcane** (5 classes)
- **Sugarcane_Healthy**
- **Sugarcane_Mosaic**
- **Sugarcane_RedRot**
- **Sugarcane_Rust**
- **Sugarcane_Yellow**

---

### 🍵 **Tea** (8 classes)
- **Tea_Anthracnose**
- **Tea_algal_leaf**
- **Tea_bird_eye_spot**
- **Tea_brown_blight**
- **Tea_gray_light**
- **Tea_healthy**
- **Tea_red_leaf_spot**
- **Tea_white_spot**

---

### 🍅 **Tomato** (9 classes)
- **Tomato_Bacterial_Spot**
- **Tomato_Early_Blight**
- **Tomato_Late_Blight**
- **Tomato_Leaf_Mold**
- **Tomato_Septoria_Leaf_Spot**
- **Tomato_Spider_Mites_Two-spotted_Spider_Mite**
- **Tomato_Target_Spot**
- **Tomato_Tomato_Yellow_Leaf_Curl_Virus**
- **Tomato_healthy**

---

### 🌾 **Wheat** (3 classes)
- **Wheat_Healthy**
- **Wheat_septoria**
- **Wheat_stripe_rust**

---

### 📊 Classification Examples

| Crop | Healthy Example | Disease Example 1 | Disease Example 2 |
|------|----------------|-------------------|-------------------|
| **Potato** | Healthy Potato Leaf | Early Blight | Late Blight |
| **Tomato** | Healthy Tomato Leaf | Bacterial Spot | Leaf Mold |
| **Rice** | Healthy Rice Plant | Rice Blast | Brown Spot |
| **Corn** | Healthy Corn Leaf | Common Rust | Gray Leaf Spot |

### 🎯 Sample Predictions

**High Confidence Detection (>80%)**
```

✅ Cauliflower_Downy_Mildew - 99.98%
✅ Sugarcane_RedRot - 91.21% confidence

```

**Low Confidence Detection (<10%)**
```

⚠️ Potato_Dry_Rot - 0.02% confidence
⚠️ Guava_Leaf_Canker - 0.04% confidence
💡 Recommendation: Image may show healthy leaf or unknown disease
```

### 🔍 How to Get Best Results

1. **Use clear, focused images** of leaves
2. **Select the correct crop type** from dropdown
3. **Ensure good lighting** and minimal background clutter
4. **Focus on diseased areas** when symptoms are visible
5. **Use high-resolution images** (minimum 224x224 pixels)

This comprehensive classification capability makes your system valuable for farmers and agricultural experts across multiple crop types! 🌱



> 🔍 **Real-world use case**: Farmers or agronomists can upload leaf images to detect early signs of disease and take preventive action.

---

## ✨ Features

- 🔄 **Transfer Learning**: Uses pre-trained ResNet-18 (ImageNet) for fast, accurate results.
- 🎯 **Multi-class Classification**: Supports over 50+ plant disease classes (e.g., *Tomato_Bacterial_spot*, *Potato_early_blight*).
- 🧪 **Data Augmentation**: Random flips, rotations, and color jitter improve generalization.
- 📈 **Validation Monitoring**: Automatically saves the best-performing model during training.
- 📦 **Modular Design**: Easy to extend for new datasets or models.
- 📂 **Checkpointing**: Saves both best and final models with metadata (class names, epoch).
- ⚙️ **Cross-platform Ready**: Works on CPU/GPU; supports Windows via safe multiprocessing.

---

## 🖼️WHY  ResNet-18 (Pretrained on ImageNet) – Transfer Learning Powerhouse ? ?
ResNet-18 is a lightweight yet powerful CNN architecture known for:

Depth without vanishing gradients: Uses skip connections (residual blocks) to train deeper networks effectively.
- Proven generalization: Trained on 1.2M ImageNet images → learns universal visual features like edges, textures, shapes.
- Fast inference: Lower parameter count (~11M) vs ResNet-50 (~25M), making it ideal for edge devices.
🔁 Transfer Learning Strategy:
We replace the final fully connected layer (fc) with a new one matching the number of plant disease classes:

Python

```bash
model.fc = nn.Linear(in_features, NUM_CLASSES)
```

This allows us to fine-tune only the last layer (and later the entire network), achieving high accuracy with minimal data.

✅ Why not a larger model like ResNet-50 or Vision Transformer?
For small-to-medium datasets (like ours), overfitting risk increases with complex models. ResNet-18 strikes the perfect balance between performance and efficiency.




# 🌿 Plant Disease Classifier - Comprehensive README

## 📁 Project Structure

Here's the recommended project structure for optimal organization and scalability:

```
plant-disease-classifier/
│
├── data/                  # Dataset files (auto-downloaded by Hugging Face)
├── models/                # Trained models (best/final checkpoints)
├── src/                   # Core Python scripts
│   ├── train_plant_disease.py  # Main training script
│   └── utils.py           # Helper functions
├── app.py                 # Streamlit web application
├── requirements.txt       # Dependency management
├── README.md              # This documentation
└── LICENSE                # MIT License
```

## 🚀 Quick Start : 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/plant-disease-classifier.git
cd plant-disease-classifier
```

### 2. Install Dependencies

## 📋 `requirements.txt` - Complete Dependency Management

Create this file in your **project root directory** (same level as `README.md`):

```text
# CORE DEEP LEARNING
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# DATA HANDLING
datasets>=2.18.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0

# VISUALIZATION & UI
matplotlib>=3.7.0
streamlit>=1.28.0

# UTILITY & EVALUATION
scikit-learn>=1.3.0
tqdm>=4.65.0
```


1. **Create virtual environment** (recommended):
   ```bash
   python -m venv plant-env
   source plant-env/bin/activate  # Linux/Mac
   plant-env\Scripts\activate    # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU Acceleration** (optional but recommended):
   ```bash
   # For NVIDIA GPU users (requires CUDA 11.8+)
   pip uninstall -y torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### ⚙️ Key Dependency Explanations

| Package | Purpose | Why Required |
|---------|---------|--------------|
| **torch/torchvision/torchaudio** | Deep learning framework | Core engine for model training/inference |
| **datasets** | Hugging Face dataset handling | Efficiently loads and processes the plant disease dataset |
| **Pillow** | Image processing | Handles all image loading/conversion operations |
| **streamlit** | Web application framework | Powers the demo UI for non-technical users |
| **scikit-learn** | Model evaluation | Provides precision/recall/F1 metrics |
| **tqdm** | Progress visualization | Shows training progress bars |



## 🚀 Getting Started

### Train the Model
```bash
python src/train_plant_disease.py
```

### Launch Web Demo
```bash
streamlit run app.py
```

### Expected Output Structure After Training
```
plant-disease-classifier/
├── models/
│   ├── best_resnet18.pth   # Best validation accuracy model
│   └── final_resnet18.pth  # Final epoch model
└── data/                   # Auto-downloaded dataset cache
```

## 📌 Critical Notes

---

## 🛠 Usage

### Train the Model


### Dataset 🗃️

* The script uses the Hugging Face datasets library to load the dataset. By default, it uses:
Python
```bash
DATASET_ID = "Saon110/bd-crop-vegetable-plant-disease-dataset"
```
- Download the dataset automatically.
- Split data into train/validation sets.
```bash
python train_plant_disease.py
```

This script will:

- Train a ResNet-18 model for 8 epochs.
- Save checkpoints to `trained_model/`.
```bash
Epoch 1/8 - LR: [0.0003]
train: 100%|████████████████████████████████████████| 313/313 [00:15<00:00, 20.87it/s]
val: 100%|███████████████████████████████████████| 35/35 [00:02<00:00, 14.50it/s]
Train loss: 0.5623 | Train acc: 0.7894 || Val loss: 0.4215 | Val acc: 0.8420
Saved best model (val_acc=0.8420) -> trained_model/best_resnet18.pth
```


## Code Structure 🧩

```bash
# Dataset Wrapper
class HFDataset(Dataset):
    def __init__(self, hf_dataset, transforms=None):
        self.ds = hf_dataset
        self.transforms = transforms

# Training Function
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # Training loop with tqdm progress bar
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        # ...

# Main Workflow
def main():
    # Load dataset, build model, train, and save
```

## Model Architecture 🏗️
The script uses ResNet18 pre-trained on ImageNet. The final fully connected layer is replaced to match the number of classes in the dataset.

``Python
``
```bash
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
```
## Training Process 🎓
1. Data Loading: Dataset is loaded and split into training and validation sets.
2. Model Initialization: ResNet18 is loaded and modified.
3. Training Loop: Train for NUM_EPOCHS.
4. Evaluation: Validate after each epoch.

> Output: `Predicted Disease: Tomato_Bacterial_spot`

---

## ⚙️ Configuration

You can customize behavior by editing the top section of `train_plant_disease.py`:

| Parameter | Default Value | Description |
|--------|---------------|------------|
| `DATASET_ID` | `"Saon110/bd-crop-vegetable-plant-disease-dataset"` | Hugging Face dataset ID |
| `OUTPUT_DIR` | `"trained_model"` | Directory to save models |
| `BATCH_SIZE` | `32` | Number of images per batch |
| `NUM_EPOCHS` | `8` | Total number of training epochs |
| `LR` | `3e-4` | Initial learning rate |
| `IMG_SIZE` | `224` | Input image size (width × height) |
| `NUM_WORKERS` | `4` | DataLoader workers (set to `0` on Windows) |
| `DEVICE` | Auto-selects `cuda` or `cpu` | Device for training |

> 🔁 Tip: Adjust `NUM_EPOCHS` or `LR` for better performance on small datasets.

---

## 📚 API Reference

### `HFDataset(Dataset)`
A wrapper for Hugging Face datasets.

#### Parameters:
- `hf_dataset`: Hugging Face dataset object (`Dataset` or `DatasetDict`)
- `transforms`: Optional `torchvision.transforms.Compose` object

#### Methods:
- `__len__()`: Returns number of samples.
- `__getitem__(idx)`: Returns `(image_tensor, label)` pair.

> ✅ Returns: `(PIL.Image)` → `(Tensor, int)` after transformation.

---

### `train_one_epoch(model, loader, optimizer, criterion, device)`
Trains one epoch.

#### Inputs:
- `model`: PyTorch model
- `loader`: `DataLoader`
- `optimizer`: Optimizer (e.g., `AdamW`)
- `criterion`: Loss function (e.g., `CrossEntropyLoss`)
- `device`: `'cuda'` or `'cpu'`

#### Returns:
- `(avg_loss: float, accuracy: float)`

---

### `evaluate(model, loader, criterion, device)`
Evaluates model on validation set.

#### Same inputs as above.

#### Returns:
- `(val_loss: float, val_acc: float)`

---

## 🔮 Future Improvements & Roadmap

1. 🔄 Use a Larger Model (e.g., EfficientNet-B0/B4 or ViT-Small)
- Try EfficientNet-B4 or Vision Transformer (ViT-Small) for higher capacity.
- Use fine-tuning with progressive unfreezing (unfreeze layers gradually).
- Leverage mixed precision training (torch.cuda.amp) for faster training.

2. 📊 Add Learning Rate Finder & Early Stopping : 
*Current Limitation: Fixed 8 epochs; may under/over-train.*

3. 📈 Integrate Weights & Biases (wandb) or TensorBoard
Goal: Visualize training metrics, model graphs, and predictions.

- Log loss, accuracy, confusion matrix, predicted vs true labels.
- Compare multiple runs (different LR, batch size, augmentations).
4. 🔄 Add Self-Supervised Pretraining (e.g., SimCLR, MoCo)


## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repo.
2. Create your feature branch (`git checkout -b feature/new-model`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-model`).
5. Open a pull request.

✅ Please ensure:
- Code follows PEP8 standards.
- Add comments where needed.
- Update `README.md` if applicable.

---

## 📄 License
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)  
![Status](https://img.shields.io/badge/Status-Stable-brightgreen?style=for-the-badge)

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

> MIT License © 2025 Subhadip Medya 

Maintained By

---

## 📞Contact

Have questions? Want to collaborate?

📧 Email: `subhadipmedya2512@gmail.com`  
🌐 GitHub: [https://github.com/Subhadip-learner](https://github.com/yourusername)  
💬 Thread: [https://www.threads.com/@qubits_subhadipxagi](https://twitter.com/yourhandle)

> Let’s grow smarter agriculture together! 🌱

---

## 🌟 Acknowledgments

- Thanks to [Hugging Face](https://huggingface.co/datasets/Saon110/bd-crop-vegetable-plant-disease-dataset) for hosting the dataset.
- Inspired by research in computer vision for precision agriculture.
- Built with ❤️ using PyTorch and open-source tools.

---

> ✅ **Ready to deploy?** Just run `streamlit run app.py ` and it will take to the website then upload the trained any of the model from models then upload the images from the data and start detecting plant diseases in seconds!

--- 
Happy coding! 🚀
