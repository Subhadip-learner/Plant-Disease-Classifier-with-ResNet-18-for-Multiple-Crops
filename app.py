import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Configure the page
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌱",
    layout="wide"
)

# ----- CONFIG -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- CLASS NAMES -----
CLASS_NAMES = [
    "Banana_Black_Pitting_or_Banana_Rust", "Banana_Crown_Rot", "Banana_Healthy", "Banana_fungal_disease",
    "Banana_leaf_Banana_Scab_Moth", "Banana_leaf_Black_Sigatoka", "Banana_leaf_Healthy", 
    "Banana_leaf__Black_Leaf_Streak", "Banana_leaf__Panama_Disease.", "Cauliflower_Bacterial_spot_rot",
    "Cauliflower_Black_Rot", "Cauliflower_Downy_Mildew", "Cauliflower_Healthy", "Corn_Blight",
    "Corn_Common_Rust", "Corn_Gray_Leaf_Spot", "Corn_Healthy", "Cotton_Aphids", "Cotton_Army worm",
    "Cotton_Bacterial blight", "Cotton_Healthy", "Guava_fruit_Anthracnose", "Guava_fruit_Healthy",
    "Guava_fruit_Scab", "Guava_fruit_Styler_end_root", "Guava_leaf_Anthracnose", "Guava_leaf_Canker",
    "Guava_leaf_Dot", "Guava_leaf_Healthy", "Guava_leaf_Rust", "Jute_Cescospora Leaf Spot",
    "Jute_Golden Mosaic", "Jute_Healthy Leaf", "Mango_Anthracnose", "Mango_Bacterial_Canker",
    "Mango_Cutting_Weevil", "Mango_Gall_Midge", "Mango_Healthy", "Mango_Powdery_Mildew",
    "Mango_Sooty_Mould", "Mango_die_back", "Papaya_Anthracnose", "Papaya_BacterialSpot",
    "Papaya_Curl", "Papaya_Healthy", "Papaya_Mealybug", "Papaya_Mite_disease", "Papaya_Mosaic",
    "Papaya_Ringspot", "Potato_Black_Scurf", "Potato_Blackleg", "Potato_Blackspot_Bruising",
    "Potato_Brown_Rot", "Potato_Common_Scab", "Potato_Dry_Rot", "Potato_Healthy_Potatoes",
    "Potato_Miscellaneous", "Potato_Pink_Rot", "Potato_Soft_Rot", "Rice_Blast", "Rice_Brownspot",
    "Rice_Tungro", "Rice_bacterial_leaf_blight", "Rice_bacterial_leaf_streak", 
    "Rice_bacterial_panicle_blight", "Rice_dead_heart", "Rice_downy_mildew", "Rice_hispa",
    "Rice_normal", "Sugarcane_Healthy", "Sugarcane_Mosaic", "Sugarcane_RedRot", "Sugarcane_Rust",
    "Sugarcane_Yellow", "Tea_Anthracnose", "Tea_algal_leaf", "Tea_bird_eye_spot", "Tea_brown_blight",
    "Tea_gray_light", "Tea_healthy", "Tea_red_leaf_spot", "Tea_white_spot", "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight", "Tomato_Late_Blight", "Tomato_Leaf_Mold", "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_Mites_Two-spotted_Spider_Mite", "Tomato_Target_Spot", 
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_healthy", "Wheat_Healthy", "Wheat_septoria",
    "Wheat_stripe_rust"
]

# ----- IMAGE TRANSFORMS -----
infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- LOAD MODEL -----
@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(CLASS_NAMES))
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        # If loading fails, try different approach
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    return model

# ----- PREDICTION FUNCTION -----
def predict_disease(model, image, crop_type, topk=3):
    """Predict disease for given image and crop type"""
    # Filter classes by crop
    crop_prefix = crop_type.capitalize() + "_"
    crop_classes = [(i, name) for i, name in enumerate(CLASS_NAMES) if name.startswith(crop_prefix)]
    
    if not crop_classes:
        return None, f"No classes found for crop: {crop_type}"
    
    # Preprocess image
    img_tensor = infer_tf(image).unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        img_logits = model(img_tensor)
        img_probs = torch.softmax(img_logits, dim=1)
    
    # Keep only crop-specific probabilities
    indices, names = zip(*crop_classes)
    crop_probs = img_probs[0, list(indices)]
    
    # Get top predictions
    topk = min(topk, len(crop_classes))
    topk_probs, topk_indices = torch.topk(crop_probs, k=topk)
    
    # Format results
    results = []
    for i, idx in enumerate(topk_indices):
        disease_name = names[idx].replace(f"{crop_type.capitalize()}_", "")
        confidence = topk_probs[i].item() * 100
        results.append({
            'disease': disease_name,
            'confidence': confidence,
            'full_name': names[idx]
        })
    
    return results, None

# ----- STREAMLIT UI -----
def main():
    st.title("🌱 Crop Disease Detection System")
    st.markdown("Upload an image of your crop leaf and select the crop type to detect potential diseases.")
    
    # Sidebar for model upload
    st.sidebar.header("Model Configuration")
    model_path = st.sidebar.file_uploader("Upload Model File (.pth)", type=['pth'])
    
    if model_path is not None:
        try:
            # Save uploaded model to temporary file
            with open("temp_model.pth", "wb") as f:
                f.write(model_path.getvalue())
            
            model = load_model("temp_model.pth")
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            return
    else:
        st.info("👈 Please upload your trained model file (.pth) in the sidebar")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose a leaf image", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Crop selection
            st.subheader("🌾 Select Crop Type")
            available_crops = sorted(list(set([name.split('_')[0] for name in CLASS_NAMES])))
            crop_type = st.selectbox("Choose the crop type:", available_crops)
            
            # Prediction button
            if st.button("🔍 Analyze Disease", type="primary"):
                with st.spinner("Analyzing image..."):
                    results, error = predict_disease(model, image, crop_type, topk=3)
                
                if error:
                    st.error(error)
                else:
                    with col2:
                        st.subheader("📊 Analysis Results")
                        
                        # Display top prediction prominently
                        top_result = results[0]
                        if top_result['confidence'] > 50:
                            st.success(f"**Primary Diagnosis:** {top_result['disease']}")
                        elif top_result['confidence'] > 20:
                            st.warning(f"**Possible Diagnosis:** {top_result['disease']}")
                        else:
                            st.info(f"**Low Confidence Detection:** {top_result['disease']}")
                        
                        st.metric("Confidence Level", f"{top_result['confidence']:.2f}%")
                        
                        # Confidence threshold warning
                        if top_result['confidence'] < 10:
                            st.warning("⚠️ Low confidence score. This might indicate:")
                            st.info("• A healthy leaf\n• An unknown disease\n• Poor image quality\n• Wrong crop selection")
                        
                        # Detailed results
                        st.subheader("Detailed Predictions")
                        results_df = pd.DataFrame(results)
                        results_df['confidence'] = results_df['confidence'].round(2)
                        st.dataframe(results_df[['disease', 'confidence']], hide_index=True)
                        
                        # Confidence visualization
                        fig, ax = plt.subplots(figsize=(10, 4))
                        diseases = [r['disease'] for r in results]
                        confidences = [r['confidence'] for r in results]
                        
                        bars = ax.barh(diseases, confidences, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                        ax.set_xlabel('Confidence (%)')
                        ax.set_title('Disease Prediction Confidence')
                        ax.bar_label(bars, fmt='%.2f%%')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Recommendations based on confidence
                        if top_result['confidence'] > 70:
                            st.success("**Recommendation:** High confidence diagnosis. Consider taking appropriate treatment measures.")
                        elif top_result['confidence'] > 30:
                            st.warning("**Recommendation:** Moderate confidence. Verify with agricultural expert if possible.")
                        else:
                            st.info("**Recommendation:** Low confidence. Please consult with agricultural expert for accurate diagnosis.")

    # Footer
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Results:")
    st.markdown("""
    - Use clear, well-lit images of leaves
    - Ensure the leaf occupies most of the image frame
    - Select the correct crop type
    - Avoid blurry or dark images
    - Focus on diseased areas of the leaf
    """)

if __name__ == "__main__":
    main()