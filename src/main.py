import cv2 as cv
from detector import LicensePlateDetector, VehicleDetector
from utils import limit_fps
import numpy as np
import easyocr


# Capture the video
video_path = '../data/traffic-flow-limited(5fps).mp4'
# video_path = '../data/traffic-flow.mp4'

cap = cv.VideoCapture(video_path)

# Load the models
vehicle_detector = VehicleDetector('../models/yolov8n.pt')
license_plate_detector = LicensePlateDetector('../models/license_plate_detector.pt')

# Load OCR
ocr_reader = easyocr.Reader(['es'])

# Save the license plates
detected_plates = [] 

# 
# line_y = 560 
# car_count = 0


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

def extract_license_plate_number(roi):
    # Apply OCR on roi 
    results = ocr_reader.readtext(roi)
    # Extract license plate number 
    license_plate_numbers = [result[1] for result in results]
    return license_plate_numbers


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    window_name = 'frame'
    window_width, window_height = 1280, 720 
    
    # Vehicle detection and bbox drawing
    v_boxes, v_scores, v_classes = car_detection(vehicle_detector)
    # print(f'v_class: {v_class}')
    v_indices = apply_NMS(v_boxes, v_scores)

    
    for i in v_indices:
        box = v_boxes[i]
        v_class = v_classes[i]
        cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        text = f'class: {v_class}' 
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, 0.5, 1)
        text_width, text_height = text_size[0] 
        cv.rectangle(frame, (box[0], box[1] - text_height - 10), (box[0] + text_width, box[1]), (0, 0, 0), -1) 
        
        cv.putText(frame, text, (box[0], box[1] - 10), cv.FONT_HERSHEY_COMPLEX,
                   0.5, (0, 255, 0), 1)


        # Count cars
        # if box[1] > line_y and box[0] > 125:
        #     print(f'CAR ON: {box[0], box[1]}')
        #     car_count += 1
            # print(f'CAR COUNT: {car_count}')


        # License plate detection and bbox drawing
        lp_boxes, lp_scores = license_plate_detection(license_plate_detector)
    
        for lp_box in lp_boxes:
            if box[0] < lp_box[0] < box[2] and box[1] < lp_box[1] < box[3]:
                # Extraer el ROI para OCR
                x1, y1, x2, y2 = lp_box
                roi = frame[y1:y2, x1:x2]
                license_plate_number = extract_license_plate_number(roi)

                if license_plate_number:
                    plate_number = license_plate_number[0]
                    detected_plates.append(plate_number)
                    text = f'plate: {plate_number}'
                    cv.putText(frame, text, (x1, y1 - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1) 
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    # print(f'BOX[1]: {box[1]}')

                    print(f'LICENSES DETECTED: {detected_plates}')
            # cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # for score in lp_scores:
            #     text = f'conf: {score:.2f}'
            #     cv.putText(frame, text, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    # Draw line to count cars
    # line_x_draw = 125
    # line_y_draw = 600 
    # cv.rectangle(frame, (line_x_draw, line_y_draw), (window_width, line_y_draw), (255, 0, 0), 3)
    # cv.putText(frame, f'car_cout: {car_count}', (10, line_y_draw - 10), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)


    # Show frames
    # cv.imshow(window_name, cv.resize(frame, (window_width, window_height)))

    cv.namedWindow("fs", cv.WND_PROP_FULLSCREEN)
    
    cv.rectangle(frame, (0, 900), (1918, 1075), (0, 0, 255), 1)

    cv.setWindowProperty("fs", cv.WND_PROP_FULLSCREEN, cv.WND_PROP_FULLSCREEN)
    cv.imshow("fs", frame)

    # Quit if 'q' pressed 
    if cv.waitKey(200) == ord('q'):
        break

    # cap.release()
    # cv.destroyAllWindows()

cap.release()
cv.destroyAllWindows()
