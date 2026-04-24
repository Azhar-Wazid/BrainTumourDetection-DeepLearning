import streamlit as st
import os
import torch
from Models.SimpleCnn import SimpleCNN
from Models.AdvancedModels import *
from Pipeline import ModelFunc, DataLoader as Dl

def build_model(model_choice):
    if model_choice == "Simple CNN":
        return SimpleCNN(inputShape=3, obj_classes=2), "simpleCNN"
    elif model_choice == "ResNet18":
        return getResnet18(), "resnet18"
    elif model_choice == "EfficientNet-B0":
        return getEfficientnet(), "efficientnet-b0"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("gpu")
else:
    device = torch.device("cpu")
    print("cpu")

st.set_page_config(page_title="Brain Tumour Detection", layout="wide")

st.sidebar.title("Controls")

model_choice = st.sidebar.selectbox(
    "Select model",
    ["Simple CNN", "ResNet18", "EfficientNet-B0"]
)

st.title("Brain Tumour Detection")
st.write(f"Selected model: **{model_choice}**")

model, modelName = build_model(model_choice)
path = f"ModelData/{modelName}/Checkpoint/bestCheckpoint.pt"

modelFunc = ModelFunc(model, device)
if os.path.exists(path):
        print("Loading saved weights")
        modelFunc.loadCheckpoint(path)
        print("Loaded weights")

model.to(device)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "tif"]
)

if uploaded_file is not None:
    image, input_tensor = Dl.preprocessSingleImage(uploaded_file, device)

    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        pred, confidence = modelFunc.predictSingleImage(model, input_tensor)

        class_names = ["normal", "tumor"]

        st.success(f"Prediction: **{class_names[pred]}**")
        st.write(f"Confidence: {confidence:.2%}")
