import torch
from PIL import Image
import torchvision.transforms as transforms
from model import create_model

# Define the class names
class_names = [
    "Tomato_Bacteria_spot",
    "Tomato_Early_bright",
    "Tomato_healthy",
    "Tomato_late_bright",
    "Tomato_leaf_mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_spider_mites_two-spotted_spider_mite",
    "Tomato_target_spot",
    "Tomato_mosaic_virus",
    "Tomato_Tomato_yellow_leaf_curl_virus"
]

# Load the model
model = create_model()
model.load_state_dict(torch.load("models/tomato_model.pth", map_location="cpu"))
model.eval()

# Define image transformation (resize to match training input)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Run test
if __name__ == "__main__":
    image_path = "sample_leaf.JPG"
    prediction = predict_image(image_path)
    print(f"Predicted disease: {prediction}")



