import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Load YOLOv8 pre-trained model
# -------------------------------
model = YOLO("yolov8n.pt")  # nano model (fast)

# -------------------------------
# Simple SORT Tracker
# -------------------------------
class SimpleTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0

    def update(self, detections):
        objects = []
        for det in detections:
            x1, y1, x2, y2 = det
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            assigned = False
            for obj_id, (px, py) in self.trackers.items():
                if abs(cx - px) < 50 and abs(cy - py) < 50:
                    self.trackers[obj_id] = (cx, cy)
                    objects.append((x1, y1, x2, y2, obj_id))
                    assigned = True
                    break

            if not assigned:
                self.trackers[self.next_id] = (cx, cy)
                objects.append((x1, y1, x2, y2, self.next_id))
                self.next_id += 1

        return objects


tracker = SimpleTracker()

# -------------------------------
# Open Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Detection
    results = model(frame, stream=True)

    detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2))

    tracked_objects = tracker.update(detections)

    # Draw Bounding Boxes & IDs
    for x1, y1, x2, y2, obj_id in tracked_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {obj_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    cv2.imshow("Object Detection & Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()