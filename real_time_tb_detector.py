import os

# Display the updated script for the user
script = """
import cv2
import torch
from torchvision import models, transforms
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
# Updated path to the model file inside the 'models' folder
model.load_state_dict(torch.load("models/tb_afb_resnet18_final.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # RGB mean
                         [0.229, 0.224, 0.225])  # RGB std
])

# Label mapping
label_map = {0: "AFB NEGATIVE", 1: "AFB POSITIVE"}

# Open video feed (0 = laptop webcam, change to 1 or 2 for USB microscope)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Unable to access camera. Try changing cv2.VideoCapture(0) to (1) or (2).")
    exit()

print("✅ Live AFB detection started — press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Convert to PIL and apply transforms
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    # Overlay result on frame
    label = f"{label_map[prediction]} ({confidence:.2f})"
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Microscope AFB Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

print(script)
