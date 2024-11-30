import warnings
import os
import platform
import pathlib
import cv2
import requests
import torch
from flask import Flask, jsonify
from config import Config  # Import the unchanged config file

warnings.filterwarnings("ignore", category=FutureWarning)

# Create Flask app and load the configuration
app = Flask(__name__)
app.config.from_object(Config)

# Adjust pathlib depending on the platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Load YOLOv5 model
yolov5_path = os.path.join(os.path.dirname(__file__), './yolov5')
model = torch.hub.load(yolov5_path, 'custom', path=app.config['WEIGHTS_PATH'], source='local')

# Open webcam (handle scenarios where webcam is unavailable, like on cloud servers)
cap = cv2.VideoCapture(0) if cv2.VideoCapture(0).isOpened() else None

# Function to send message via LINE Notify
def send_line_notify(message):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {app.config['LINE_NOTIFY_TOKEN']}"}
    data = {"message": message}
    response = requests.post(url, headers=headers, data=data)
    return response.status_code

# API for object detection
@app.route('/api/detect', methods=['GET'])
def detect_objects():
    if cap is None or not cap.isOpened():
        return jsonify({"error": "Webcam not accessible"}), 500

    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    # Perform detection using YOLO
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get detection results as a DataFrame

    detected_objects = []
    for _, row in detections.iterrows():
        if row['confidence'] > app.config['CONFIDENCE_THRESHOLD']:
            detected_objects.append({
                "label": row['name'],
                "confidence": row['confidence']
            })

            # Send notification via LINE Notify
            message = f"Detected: {row['name']} with confidence {row['confidence']:.2f}"
            send_line_notify(message)

    return jsonify({"objects": detected_objects})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))  # Use Render's assigned port or default to 5000
    app.run(debug=False, host='0.0.0.0', port=port)
