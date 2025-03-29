import cv2
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, Response
from torchvision import models
import torch.nn as nn
from PIL import Image
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# Global variables
cap = None
stop_streaming = False
capture_thread = None

# Load the models for each task
tasks = {
    "face_recognition": "models/face_recognition_model.pth",
    "mask_detection": "models/mask_detection_model.pth",
    "lighting": "models/lighting_model.pth",
    "crowd_density": "models/crowd_density_model.pth",
    "suspicious_object": "models/suspicious_object_model.pth",
    "animal_intrusion": "models/animal_intrusion_model.pth",
    "motion_detection": "models/motion_detection_model.pth"
}

# Load pre-trained ResNet models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dict = {}

def load_models():
    print("Loading models...")
    for task, model_path in tasks.items():
        print(f"Loading model for task: {task} from {model_path}")
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Adjusting for binary classification
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        models_dict[task] = model
    print("Models loaded successfully.")

# Image transformation for the models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ResNet normalization
])

def detect_tasks(frame):
    print("Detecting tasks...")
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Store results of each task
    results = {}

    # Run inference for each model
    for task, model in models_dict.items():
        print(f"Running inference for {task}...")
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            result = "Unknown" if predicted.item() == 0 else "Known"  # Adjust as per your labels
            results[task] = result

    print("Task detection completed.")
    return results

def generate_video_stream():
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run all models on the captured frame
        mask_prediction = mask_model.predict(frame)
        lighting_prediction = lighting_model.predict(frame)
        crowd_prediction = crowd_model.predict(frame)
        suspicious_prediction = suspicious_model.predict(frame)
        animal_prediction = animal_model.predict(frame)
        motion_prediction = motion_model.predict(frame)

        # Prepare text overlay
        overlay_text = f"Mask: {mask_prediction}, Lighting: {lighting_prediction}\n" \
                       f"Crowd: {crowd_prediction}, Suspicious: {suspicious_prediction}\n" \
                       f"Animal: {animal_prediction}, Motion: {motion_prediction}"

        # Draw overlay text on frame
        cv2.putText(frame, overlay_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    print("Rendering index page.")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    print("Providing video feed...")
    # Start the video stream
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Global variable to track whether the capture thread is already running
capture_thread = None

@app.route('/start_stream')
def start_stream():
    global cap, capture_thread
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
        if not cap.isOpened():
            return "Error: Could not access the camera."
        print("Starting capture thread...")
        capture_thread = threading.Thread(target=start_video_capture)
        capture_thread.start()
        return "Streaming started."
    else:
        print("Streaming already in progress.")
        return "Streaming already in progress."

@app.route('/stop_stream')
def stop_stream():
    global cap, capture_thread
    if cap is not None and cap.isOpened():
        print("Stopping video stream...")
        # Stop the video capture and release the camera
        cap.release()
        cap = None  # Reset the cap object
        if capture_thread is not None:
            capture_thread.join()  # Wait for the capture thread to finish
            capture_thread = None  # Reset capture thread
        return "Streaming stopped."
    else:
        return "No active streaming to stop."

def start_video_capture():
    global cap
    while cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        # Process the frame (e.g., send it to clients via Flask)
        # Add your frame processing logic here

    print("Capture thread stopped.")

if __name__ == '__main__':
    # Load all models before starting the application
    load_models()
    print("Starting Flask app...")
    app.run(debug=True, threaded=True)
