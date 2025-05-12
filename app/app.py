import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

@st.cache_resource
def load_model():
    # 1) Instantiate the ResNet18 architecture
    model = resnet18(pretrained=False)
    # 2) Replace its final fc layer for 2 output classes
    num_feats = model.fc.in_features
    model.fc = torch.nn.Linear(num_feats, 2)
    # 3) Load your weights (assumes you saved state_dict)
    state_dict = torch.load('models/tb_afb_resnet18_final.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

st.title("TB / Malaria Detection Demo")
st.write("Upload a microscope slide image to detect TB or malaria bacilli.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded image", use_column_width=True)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits[0], dim=0)
        pred = torch.argmax(probs).item()

    labels = ["Negative","Positive"]
    st.markdown(f"## Prediction: **{labels[pred]}**")
    st.markdown(f"**Confidence:** {probs[pred]:.2f}")

