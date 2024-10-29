import cv2 as cv
from detector import LicensePlateDetector, VehicleDetector
from utils import limit_fps
import numpy as np


# Load the model
detector = LicensePlateDetector("../models/license_plate_detector.pt")

# Capture the video
# video_path = limit_fps('../data/traffic-flow.mp4', None, 5)
# video_path = "../data/traffic-flow.mp4"
video_path = '../data/traffic-flow-limited(5fps).mp4'

cap = cv.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    window_name = 'frame'
    window_width, window_height = 800, 500 

    # license_results = detector.detect(frame)
    # print(results)
    
    # for box in results[0].boxes:
    #     x1, y1, x2, y2 = map(int, box.xyxy[0])
    #     class_id = int(box.cls[0])
    #     confidence = float(box.conf[0])

    #     if class_id == 0 and confidence > 0.5:
    #         cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv.putText(frame, f'Conf: {confidence:.2f}', (x1, y1 - 10),
    #                    cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)


    vehicle_d = VehicleDetector('../models/yolov8n.pt')
    vehicle_results = vehicle_d.detect(frame)
    # print(vehicle_results) 

    boxes = []
    scores = []

    for box in vehicle_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append([x1, y1, x2, y2])
        scores.append(box.conf[0])
        # cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    boxes = np.array(boxes)
    scores = np.array(scores)
    # Apply Non-maxium suppresion
    indices = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=0.4)

    # NMS
    for i in indices:
        box = boxes[i]
        cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # cv.imshow(window_name, frame)
    cv.imshow(window_name, cv.resize(frame, (window_width, window_height)))
    # cv.resize(frame, (window_width, window_height))
    if cv.waitKey(200) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
