import cv2 as cv
from detector import LicensePlateDetector, VehicleDetector
from utils import limit_fps
import numpy as np


# Capture the video
video_path = '../data/traffic-flow-limited(5fps).mp4'

cap = cv.VideoCapture(video_path)


# Load the models
vehicle_detector = VehicleDetector('../models/yolov8n.pt')
license_plate_detector = LicensePlateDetector('../models/license_plate_detector.pt')


def car_detection(model):
    """
    Using yolov8n.pt model to detect vehicles and get their boxes and scores

    Return:
        boxes -> 
        scores -> 
    """
    vehicle_results = model.detect(frame)
    
    boxes = []
    scores = []

    vehicle_classes = [2, 3, 5, 7]

    for box in vehicle_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0]) 
        
        if class_id in vehicle_classes and confidence > 0.5:
            boxes.append([x1, y1, x2, y2])
            scores.append(box.conf[0])
    
    return boxes, scores


def license_plate_detection(model):
    license_results = model.detect(frame)

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

def apply_NMS(boxes, scores):
    """
        Apply non max suppression technique to select the best bounding boxes 
        out of a set of overlapping boxes 

        Return:
            indices ->
    """
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Apply Non-maxium suppresion
    indices = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                              score_threshold=0.5, nms_threshold=0.4)

    return indices


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    window_name = 'frame'
    window_width, window_height = 1024, 720 
    
    # Vehicle detection and bbox drawing
    v_boxes, v_scores = car_detection(vehicle_detector)
    v_indices = apply_NMS(v_boxes, v_scores)

    for i in v_indices:
        box = v_boxes[i]
        cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # License plate detection and bbox drawing
    lp_boxes, lp_scores = license_plate_detection(license_plate_detector)
    
    for box in lp_boxes:
        cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        for score in lp_scores:
            cv.putText(frame, f'conf: {score:.2f}', (box[0], box[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    
    # Show frames
    cv.imshow(window_name, cv.resize(frame, (window_width, window_height)))

    # Quit if 'q' pressed 
    if cv.waitKey(200) == ord('q'):
        break

    # cap.release()
    # cv.destroyAllWindows()

cap.release()
cv.destroyAllWindows()
