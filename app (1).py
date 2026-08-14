import streamlit as st
from fastai.vision.core import PILImage
from fastai.learner import load_learner
import torch
import torchvision.transforms as transforms

# Set the title of the app
st.title("Derma: An AI-Based Skin Disease Screening System")
st.write("by Ruengsit Matachaiyasit & Teerapat Sittichottithikun")

# Load the FastAI model
@st.cache_resource
def load_model():
    model = load_learner("Skin_disease (1).pkl")
    return model

model = load_model()

# Define image transformations (optional, depending on your model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# File uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = PILImage.create(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Make prediction using FastAI's model
    pred, pred_idx, probs = model.predict(image)

    pred_idx = pred_idx.item()

    # Display the prediction
    st.write(f"Prediction: **{pred}**")
    st.write(f"Probability: **{probs[pred_idx]:.4f}*100**")
else:
    st.write("Please upload an image.")
