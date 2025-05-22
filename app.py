import streamlit as st
import torch
import torch.nn as nn
from neural_networks.simple_cnn import SimpleCNN
from neural_networks.tuning_simple_cnn import TuningSimpleCNN
from neural_networks.tuning_simple_cnn_bn import TuningSimpleCNNWithBN
from torchvision import models, transforms
from PIL import Image

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ]
)


def predict_image(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        if (
            isinstance(output, torch.Tensor) and output.size(1)
            if len(output.shape) > 1
            else 1 == 1
        ):
            confidence = output.item()
            predicted_class = 1 if confidence >= 0.5 else 0
        else:
            confidence, predicted_class = torch.max(output, 1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()
    return predicted_class, confidence


if "model_dict" not in st.session_state:
    model_dict = {}

    model_dict["SimpleCNN"] = SimpleCNN().to(device)
    model_dict["TunedCNN"] = TuningSimpleCNN().to(device)
    model_dict["TunedCNNWithBN"] = TuningSimpleCNNWithBN().to(device)

    model_dict["ResNet"] = models.resnet18(weights=None)
    model_dict["ResNet"].fc = nn.Sequential(
        nn.Linear(model_dict["ResNet"].fc.in_features, 1), nn.Sigmoid()
    )
    model_dict["ResNet"].load_state_dict(
        torch.load("models/resnet_transfer.pt", map_location=device)
    )
    model_dict["ResNet"] = model_dict["ResNet"].to(device)

    model_dict["SimpleCNN"].load_state_dict(
        torch.load("models/cat_dog_model_v1.pt", map_location=device)
    )
    model_dict["TunedCNN"].load_state_dict(
        torch.load("models/optuna_best_model_final.pt", map_location=device)
    )
    model_dict["TunedCNNWithBN"].load_state_dict(
        torch.load("models/optuna_bn.pt", map_location=device)
    )

    for model in model_dict.values():
        model.eval()

    st.session_state.model_dict = model_dict

# Interface
st.title("🐱 Cat vs Dog 🐶 Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with col2:
        st.subheader("Model Predictions:")
        for model_name, model in st.session_state.model_dict.items():
            predicted_class, raw_confidence = predict_image(model, input_tensor, device)
            pred = "Dog" if predicted_class == 1 else "Cat"

            # Calculate proper confidence value (0.0-1.0)
            # For binary classification where output is a probability for class 1 (Dog)
            confidence = raw_confidence if pred == "Dog" else 1.0 - raw_confidence

            # Format confidence as percentage
            st.write(f"**{model_name}**: {pred} ({confidence:.2f})")
