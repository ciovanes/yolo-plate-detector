from ultralytics import YOLO

class LicensePlateDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        result = self.model(frame, verbose=False)
        return result
    

class VehicleDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect(self, frame):
        result = self.model(frame, verbose=False)
        return result