import cv2 as cv
from detector import LicensePlateDetector, VehicleDetector
from utils import * 
import numpy as np
from saver import LicensePlateSaver


class TrafficFlowAnalyzer:

    def __init__(self, video_path, vehicle_model, license_plate_model):
        self.video_path = video_path
        self.vehicle_model = vehicle_model
        self.license_plate_model = license_plate_model
        self.detected_plates = set()
        self.saver = LicensePlateSaver()


    def vehicle_detection(self, frame):
        """
        Using yolov8n.pt model to detect vehicles and get their boxes and scores

        Return:
            boxes -> 
            scores -> 
        """
        vehicle_results = self.vehicle_model.detect(frame)

        boxes = []
        scores = []
        classes = []

        vehicle_classes = [2, 3, 5, 7]

        vehicle_dict = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        for box in vehicle_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0]) 

            if class_id in vehicle_classes and confidence > 0.5:
                # print(f'class: {class_id}, {vehicle_dict.get(class_id)}')
                boxes.append([x1, y1, x2, y2])
                # Convert the confidence score tensor to a CPU-based float before appending.
                scores.append(box.conf[0].cpu()) 
                classes.append(vehicle_dict.get(class_id)) 

        return boxes, scores, classes 


    def draw_vehicle_bbox(self, frame, box, v_class):
        cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        text = f'class: {v_class}' 
        draw_text(frame, text, box[0], box[1], (box[0], box[1] - 10), 
                  (0, 0, 0), (0, 255, 0))


    def license_plate_detection(self, frame):
        license_results = self.license_plate_model.detect(frame)
    
        boxes = []
        scores = []
    
        for box in license_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
    
            if class_id == 0 and confidence > 0.5:
                boxes.append([x1, y1, x2, y2])
                scores.append(box.conf[0])
    
        return boxes, scores 
    

    def draw_licence_plate_bbox(self, frame, box, plate_number):
        text = f'plate: {plate_number}'
        cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        draw_text(frame, text, box[0], box[1], (box[0], box[1] - 10), (0, 0, 0), (0, 255, 255))


    def add_to_detected_plates(self, license_plate):
        # Check current size
        current_size = len(self.detected_plates)

        # Add non duplicates license plates
        self.detected_plates.add(license_plate)

        if len(self.detected_plates) > current_size:
            return True

        return False 


    def run(self):
        cap = cv.VideoCapture(self.video_path)
        
        # detected_plates = set()

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
    
            window_name = 'frame'

            cv.namedWindow(window_name, cv.WINDOW_NORMAL)

            # screen_width = cv.getWindowImageRect(window_name)[2]
            # screen_height = cv.getWindowImageRect(window_name)[3]
            # screen_width = 1280
            # screen_height = 720 
            # cv.resizeWindow(window_name, screen_width, screen_height)

            # Get car detection (bbox, confidence and class)
            v_boxes, v_scores, v_classes = self.vehicle_detection(frame)
            # Apply NMS to bbox
            v_indices = apply_NMS(v_boxes, v_scores)

            # Recorrer todos los bbox 
            for i in v_indices:
                box = v_boxes[i]
                v_class = v_classes[i]
                self.draw_vehicle_bbox(frame, box, v_class)
                
                # Detect license plate on side of vehicle
                lp_boxes, lp_scores = self.license_plate_detection(frame)
                license_plate_number = None

                for lp_box in lp_boxes:
                    # Si esta dentro del coche
                    if box[0] < lp_box[0] < box[2] and box[1] < lp_box[1] < box[3]:
                        # Extraer el ROI para OCR
                        x1, y1, x2, y2 = lp_box
                        roi = frame[y1:y2, x1:x2]
                        license_plate_number = self.license_plate_model.extract_license_plate_number(roi)
                        # print(license_plate_number)

                    if license_plate_number and is_valid_license_plate(license_plate_number[0]):
                        license_plate = format_license_plate(license_plate_number[0])
                        box = [x1, y1, x2, y2]
                        self.draw_licence_plate_bbox(frame, box, license_plate)

                        if self.add_to_detected_plates(license_plate):
                            self.saver.save(license_plate=license_plate)


            # Quit if 'q' pressed 
            if cv.waitKey(200) == ord('q'):
                break

            cv.imshow(window_name, frame)
        
            # cap.release()
            # cv.destroyAllWindows()


        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':

    # video_path = limit_fps(src='../data/traffic-flow.mp4', fps=5) 
    video_path = '../data/traffic-flow-limited(5fps).mp4'
    vehicle_detector = VehicleDetector('../models/yolov8n.pt')
    license_plate_detector = LicensePlateDetector('../models/license_plate_detector.pt')

    analyzer = TrafficFlowAnalyzer(
        video_path,
        vehicle_detector,
        license_plate_detector
    ) 

    analyzer.run()
