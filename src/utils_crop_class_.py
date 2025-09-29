import torch
from torchvision import transforms, models
from PIL import Image
import argparse

# ----- CONFIG -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- CLASS NAMES -----
CLASS_NAMES = [
    "Banana_Black_Pitting_or_Banana_Rust",
    "Banana_Crown_Rot",
    "Banana_Healthy",
    "Banana_fungal_disease",
    "Banana_leaf_Banana_Scab_Moth",
    "Banana_leaf_Black_Sigatoka",
    "Banana_leaf_Healthy",
    "Banana_leaf__Black_Leaf_Streak",
    "Banana_leaf__Panama_Disease.",
    "Cauliflower_Bacterial_spot_rot",
    "Cauliflower_Black_Rot",
    "Cauliflower_Downy_Mildew",
    "Cauliflower_Healthy",
    "Corn_Blight",
    "Corn_Common_Rust",
    "Corn_Gray_Leaf_Spot",
    "Corn_Healthy",
    "Cotton_Aphids",
    "Cotton_Army worm",
    "Cotton_Bacterial blight",
    "Cotton_Healthy",
    "Guava_fruit_Anthracnose",
    "Guava_fruit_Healthy",
    "Guava_fruit_Scab",
    "Guava_fruit_Styler_end_root",
    "Guava_leaf_Anthracnose",
    "Guava_leaf_Canker",
    "Guava_leaf_Dot",
    "Guava_leaf_Healthy",
    "Guava_leaf_Rust",
    "Jute_Cescospora Leaf Spot",
    "Jute_Golden Mosaic",
    "Jute_Healthy Leaf",
    "Mango_Anthracnose",
    "Mango_Bacterial_Canker",
    "Mango_Cutting_Weevil",
    "Mango_Gall_Midge",
    "Mango_Healthy",
    "Mango_Powdery_Mildew",
    "Mango_Sooty_Mould",
    "Mango_die_back",
    "Papaya_Anthracnose",
    "Papaya_BacterialSpot",
    "Papaya_Curl",
    "Papaya_Healthy",
    "Papaya_Mealybug",
    "Papaya_Mite_disease",
    "Papaya_Mosaic",
    "Papaya_Ringspot",
    "Potato_Black_Scurf",
    "Potato_Blackleg",
    "Potato_Blackspot_Bruising",
    "Potato_Brown_Rot",
    "Potato_Common_Scab",
    "Potato_Dry_Rot",
    "Potato_Healthy_Potatoes",
    "Potato_Miscellaneous",
    "Potato_Pink_Rot",
    "Potato_Soft_Rot",
    "Rice_Blast",
    "Rice_Brownspot",
    "Rice_Tungro",
    "Rice_bacterial_leaf_blight",
    "Rice_bacterial_leaf_streak",
    "Rice_bacterial_panicle_blight",
    "Rice_dead_heart",
    "Rice_downy_mildew",
    "Rice_hispa",
    "Rice_normal",
    "Sugarcane_Healthy",
    "Sugarcane_Mosaic",
    "Sugarcane_RedRot",
    "Sugarcane_Rust",
    "Sugarcane_Yellow",
    "Tea_Anthracnose",
    "Tea_algal_leaf",
    "Tea_bird_eye_spot",
    "Tea_brown_blight",
    "Tea_gray_light",
    "Tea_healthy",
    "Tea_red_leaf_spot",
    "Tea_white_spot",
    "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_Mites_Two-spotted_Spider_Mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_healthy",
    "Wheat_Healthy",
    "Wheat_septoria",
    "Wheat_stripe_rust"
]

# ----- IMAGE TRANSFORMS -----
infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- ARGUMENTS -----
parser = argparse.ArgumentParser(description="Predict crop disease from leaf image")
parser.add_argument("image", type=str, help="Path to leaf image")
parser.add_argument("--crop", type=str, required=True, help="Crop type (Potato, Tomato, Rice, etc.)")
parser.add_argument("--topk", type=int, default=1, help="Top-k predictions")
parser.add_argument("--model", type=str, default="best_resnet18.pth", help="Path to trained model checkpoint")
args = parser.parse_args()

# ----- LOAD MODEL -----
model = models.resnet18(weights=None)  # no pretrained, use your trained weights
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, len(CLASS_NAMES))
checkpoint = torch.load(args.model, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# ----- LOAD IMAGE -----
img = Image.open(args.image).convert("RGB")
img_tensor = infer_tf(img).unsqueeze(0).to(DEVICE)

# ----- FILTER CLASSES BY CROP -----
crop_prefix = args.crop.capitalize() + "_"  # e.g., "Potato_"
crop_classes = [(i, name) for i, name in enumerate(CLASS_NAMES) if name.startswith(crop_prefix)]

if not crop_classes:
    print(f"No classes found for crop: {args.crop}")
    exit()

indices, names = zip(*crop_classes)
img_logits = model(img_tensor)
img_probs = torch.softmax(img_logits, dim=1)

# Keep only crop-specific probabilities
crop_probs = img_probs[0, list(indices)]
topk = min(args.topk, len(crop_classes))
topk_probs, topk_indices = torch.topk(crop_probs, k=topk)

# ----- PRINT RESULTS -----
print("\nPredictions for crop:", args.crop)
for i, idx in enumerate(topk_indices):
    print(f"{names[idx]} — {topk_probs[i]*100:.2f}%")
