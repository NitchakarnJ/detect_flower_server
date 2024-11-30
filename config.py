import os

class Config:
    # LINE Notify Token (set it through environment variables for security)
    LINE_NOTIFY_TOKEN = os.getenv("LINE_NOTIFY_TOKEN", "5csNQjR898Iev5G5ENGXP3q7QmNr6l5ZDZBvjHhwLhf")

    # YOLOv5 Model Path (could be set dynamically if you need to change it)
    WEIGHTS_PATH = "flower_detection_model.pt"  # Replace with your correct path or set dynamically
    
    # Confidence Threshold for Object Detection
    CONFIDENCE_THRESHOLD = 0.60  # Filter out detections with lower confidence
