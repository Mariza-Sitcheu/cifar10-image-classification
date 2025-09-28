import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import CustomCNN
from src.preprocess import load_cifar10_data

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN()
model.load_state_dict(torch.load("src/models/custom_cnn.pth", weights_only=True))
model = model.to(device)
model.eval()

# Load classes
_, _, classes = load_cifar10_data()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

st.title("CIFAR-10 Image Classification")
st.write("Upload an image to classify it using a custom CNN.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = classes[predicted.item()]

    st.write(f"Predicted Class: **{pred_class}**")