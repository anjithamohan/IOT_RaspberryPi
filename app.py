import cv2
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from torchvision import models
import torch.nn as nn
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model paths
tasks = {
    "face_recognition": "models/face_recognition_model.pth",
    "mask_detection": "models/mask_detection_model.pth",
    "lighting": "models/lighting_model.pth",
    "crowd_density": "models/crowd_density_model.pth",
    "suspicious_object": "models/suspicious_object_model.pth",
    "animal_intrusion": "models/animal_intrusion_model.pth",
    "motion_detection": "models/motion_detection_model.pth"
}

# Load models
models_dict = {}
def load_models():
    print("Loading models...")
    for task, model_path in tasks.items():
        print(f"Loading model for {task} from {model_path}")
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models_dict[task] = model
    print("Models loaded successfully.")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    image = Image.open(file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    results = {}
    for task, model in models_dict.items():
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            results[task] = "Known" if predicted.item() == 1 else "Unknown"
    
    return jsonify(results)

if __name__ == '__main__':
    load_models()
    app.run(debug=True, threaded=True)
