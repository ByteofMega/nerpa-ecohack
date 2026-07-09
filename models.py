from ultralytics import YOLO

SATELLITE_WEIGHTS = "weights/satellite_best.pt"
DRONE_WEIGHTS = "weights/drone_best.pt"


class ModelManager:
    def __init__(self, satellite_weights=SATELLITE_WEIGHTS, drone_weights=DRONE_WEIGHTS):
        self.satellite_model = YOLO(satellite_weights)
        self.drone_model = YOLO(drone_weights)

    def detect_satellite(self, img_bgr, conf=0.4):
        results = self.satellite_model.predict(img_bgr, conf=conf, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            boxes.append([int(x1), int(y1), int(x2), int(y2), cls_id])
        return boxes

    def detect_drone(self, frame_bgr, conf=0.4):
        results = self.drone_model.predict(frame_bgr, conf=conf, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            boxes.append([int(x1), int(y1), int(x2), int(y2), cls_id])
        return boxes
