import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, jsonify
import os
import platform
import pathlib
import torch
import cv2
import requests  # For sending messages via LINE Notify
from config import Config  # Import the config file

# Create Flask app and load the configuration
app = Flask(__name__)
app.config.from_object(Config)  # Use Config class from the config.py

# Adjust pathlib depending on the platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Load YOLOv5 model

model = torch.hub.load('./yolov5', 'custom', path=app.config['WEIGHTS_PATH'], source='local')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to send message via LINE Notify
def send_line_notify(message):
    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": f"Bearer {app.config['LINE_NOTIFY_TOKEN']}"
    }
    data = {"message": message}
    response = requests.post(url, headers=headers, data=data)
    return response.status_code

# API for object detection
@app.route('/api/detect', methods=['GET'])
def detect_objects():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    # Perform detection using YOLO
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get detection results as a DataFrame

    detected_objects = []
    for _, row in detections.iterrows():
        if row['confidence'] > app.config['CONFIDENCE_THRESHOLD']:  # Use the threshold from config
            detected_objects.append({
                "label": row['name'],
                "confidence": row['confidence']
            })

            # Send notification via LINE Notify
            message = f"Detected: {row['name']} with confidence {row['confidence']:.2f}"
            send_line_notify(message)

    return jsonify({"objects": detected_objects})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
