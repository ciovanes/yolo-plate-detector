import cv2 as cv
from detector import LicensePlateDetector
from utils import limit_fps

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

    results = detector.detect(frame)
    # print(results)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id == 0:
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f'Conf: {confidence:.2f}', (x1, y1 - 10),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    cv.imshow("Deteccion de matriculas", frame)
    if cv.waitKey(200) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
