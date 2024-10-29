from ultralytics import YOLO

class LicensePlateDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        result = self.model(frame)
        return result