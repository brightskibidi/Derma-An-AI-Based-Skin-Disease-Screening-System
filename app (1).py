import streamlit as st
from fastai.vision.core import PILImage
from fastai.learner import load_learner
import torch
import torchvision.transforms as transforms
#disease info
disease_info = {
    "cellulitis": {
        "description": "A bacterial infection of the deeper layers of the skin.",
        "causes": [
            "Bacterial infection, commonly through a break in the skin",
            "Cuts, wounds, or insect bites can provide an entry point for bacteria"
        ],
        "what_to_do": [
            "Seek medical evaluation, especially if the affected area is rapidly spreading",
            "Keep the affected area clean",
            "Follow medical advice and prescribed treatment"
        ]
    },

    "impetigo": {
        "description": "A contagious bacterial skin infection that commonly affects the surface of the skin.",
        "causes": [
            "Usually caused by Staphylococcus or Streptococcus bacteria",
            "Can spread through direct contact or contaminated objects"
        ],
        "what_to_do": [
            "Keep the affected area clean",
            "Avoid scratching or touching the affected area",
            "Seek medical advice for appropriate treatment"
        ]
    },

    "athlete's foot": {
        "description": "A fungal infection that commonly affects the skin of the feet.",
        "causes": [
            "Fungal infection",
            "Warm and moist environments can promote fungal growth",
            "Can spread through contaminated floors, shoes, or towels"
        ],
        "what_to_do": [
            "Keep the feet clean and dry",
            "Change socks regularly",
            "Avoid sharing towels, socks, or footwear",
            "Seek medical advice if symptoms persist or worsen"
        ]
    },

    "nail fungus": {
        "description": "A fungal infection affecting the fingernails or toenails.",
        "causes": [
            "Fungal infection",
            "Can spread through contact with contaminated surfaces or objects"
        ],
        "what_to_do": [
            "Keep nails clean and dry",
            "Avoid sharing nail clippers or other personal items",
            "Seek medical advice for appropriate treatment"
        ]
    },

    "ringworm": {
        "description": "A fungal infection that can cause circular or ring-shaped skin lesions.",
        "causes": [
            "Fungal infection",
            "Can spread through contact with infected people, animals, or contaminated objects"
        ],
        "what_to_do": [
            "Keep the affected area clean and dry",
            "Avoid sharing towels, clothing, or personal items",
            "Seek medical advice if the condition does not improve"
        ]
    },

    "chickenpox": {
        "description": "A contagious viral infection that commonly causes an itchy rash and fluid-filled spots.",
        "causes": [
            "Caused by the varicella-zoster virus",
            "Spreads mainly through respiratory droplets and close contact"
        ],
        "what_to_do": [
            "Avoid close contact with others while contagious",
            "Avoid scratching the rash",
            "Seek medical advice when necessary"
        ]
    },

    "shingles": {
        "description": "A viral infection that causes a painful skin rash and occurs when the varicella-zoster virus becomes reactivated.",
        "causes": [
            "Reactivation of the varicella-zoster virus",
            "The virus can remain inactive in the body after chickenpox"
        ],
        "what_to_do": [
            "Seek medical evaluation promptly",
            "Avoid close contact with people who have not had chickenpox or vaccination",
            "Follow medical advice and prescribed treatment"
        ]
    }
}
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
    confidence = float(probs[pred_idx]) * 100

    st.write(f"### Prediction: {pred}")
    st.write(f"**Model Confidence: {confidence:.2f}%**")

    info = disease_info.get(str(pred))

    if info:
        st.write("### About this condition")
        st.write(info["description"])

        st.write("### Possible causes")
        for cause in info["causes"]:
            st.write(f"- {cause}")

        st.write("### What you can do")
        for action in info["what_to_do"]:
            st.write(f"- {action}")
        st.warning(
            "This result is an AI-based preliminary screening and is not a medical diagnosis. "
            "Please consult a qualified healthcare professional for proper evaluation."
)
    else:
        st.write("Please upload an image.")
