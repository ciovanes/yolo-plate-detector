from ultralytics import YOLO
import easyocr


class LicensePlateDetector:

    """
        Class to detect license plates
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ocr_reader = easyocr.Reader(['en'])


    def detect(self, frame):
        result = self.model(frame, verbose=False)
        return result


    def extract_license_plate_number(self, roi):
        # Apply OCR on roi 
        results = self.ocr_reader.readtext(roi)
        return  [result[1] for result in results]


class VehicleDetector:

    """
        Class to detect vehicles
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    
    def detect(self, frame):
        result = self.model(frame, verbose=False)
        return result