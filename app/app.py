import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Cache the loaded model so it’s only loaded once per session
@st.cache_resource
def load_model():
    model = torch.load('models/tb_afb_resnet18_final.pth', map_location='cpu')
    model.eval()
    return model

model = load_model()

st.title("TB / Malaria Detection Demo")
st.write("Upload a microscope slide image to detect TB or malaria bacilli.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded image', use_column_width=True)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output[0], dim=0)
        pred = torch.argmax(probs).item()

    labels = ['Negative', 'Positive']
    st.markdown(f"## Prediction: **{labels[pred]}**")
    st.markdown(f"**Confidence:** {probs[pred]:.2f}")
