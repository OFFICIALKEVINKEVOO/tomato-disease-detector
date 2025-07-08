import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import json
from model import TomatoCNN

# Load class names
with open("models/class_names.json") as f:
    class_names = json.load(f)

# Load model
model = TomatoCNN(num_classes=len(class_names))
model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# --- Define treatment tips ---
treatments = {
    "Tomato_healthy": "✅ No treatment needed. Keep monitoring the plant and ensure healthy growing conditions.",
    "Tomato_Late_blight": "🧪 Use fungicides like Chlorothalonil or copper-based sprays. Remove affected leaves. Avoid overhead watering.",
    "Tomato_Septoria_leaf_spot": "🌱 Apply copper-based fungicides weekly. Prune lower leaves. Improve airflow between plants.",
}

recommendations = """
**🌿 General Recommendations:**
- Ensure proper crop rotation every season.
- Water at the base of the plant to avoid wetting the leaves.
- Keep the garden free from infected plant debris.
- Monitor plants weekly for early signs of infection.
"""

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🍅 Predict", "ℹ️ About", "📘 How to Use", "🧠 Model Info"])

# --- Tab 1: Prediction ---
with tab1:
    st.title("🍅 Tomato Leaf Disease Detector")
    st.write("Upload an image or take a photo to predict tomato leaf diseases.")

    option = st.radio("Choose input method:", ("📁 Upload Image", "📷 Take Photo"))

    image = None
    if option == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    elif option == "📷 Take Photo":
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")

    if image:
        st.image(image, caption="Input Image", use_column_width=True)
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probs, 0)
            confidence = confidence.item() * 100
            label = class_names[predicted.item()]

        if confidence < 99:
            st.error("🚫 This image is *not* recognized as a valid tomato leaf.")
        else:
            st.success(f"🌿 Prediction: **{label}**\nConfidence: **{confidence:.2f}%**")

            # Show treatment tip
            if label in treatments:
                st.info(f"💊 **Treatment Tip:** {treatments[label]}")

            # Always show general tips
            st.markdown("---")
            st.markdown(recommendations)

# --- Tab 2: About ---
with tab2:
    st.title("ℹ️ About This App")
    st.markdown("""
    This Tomato Leaf Disease Detector helps farmers and agricultural experts:

    - Identify common tomato diseases using deep learning 🌱
    - Offer accurate predictions with over **99% confidence** ✅
    - Suggest treatments for affected leaves 💊

    **Built by KELVIN KYANULA.**
    """)

# --- Tab 3: How to Use ---
with tab3:
    st.title("📘 How to Use")
    st.markdown("""
    1. Go to the **Predict** tab.
    2. Choose between uploading an image or using your camera.
    3. Wait for the model to analyze and give a result.
    4. If it's a disease, you’ll also get a **treatment tip**!

    > 🔍 For best results, use **clear, close-up images** of **only tomato leaves**.
    """)

# --- Tab 4: Model Info ---
with tab4:
    st.title("🧠 Model Info")
    st.markdown(f"""
    - **Model Type:** Custom Convolutional Neural Network (CNN)
    - **Framework:** PyTorch
    - **Image Size:** 128x128
    - **Classes:** {", ".join(class_names)}
    - **Accuracy:** >99% on clean validation data

    📁 **Model Path:** `models/model.pth`  
    📄 **Class Names File:** `models/class_names.json`
    """)
