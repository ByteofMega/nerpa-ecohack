import numpy as np
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


# ================== ЗАГРУЗКА МОДЕЛЕЙ ==================

class ModelManager:
    """
    Хранит две модели YOLO:
    - satellite_model: детекция мусорных пятен / крупной структуры на снимке сверху
    - drone_model: детекция мусора/водорослей по кадру с дрона
    """

    def __init__(self,
                 satellite_weights: str = "satellite_best.pt",
                 drone_weights: str = "drone_best.pt"):
        self.satellite_model = YOLO(satellite_weights)
        self.drone_model = YOLO(drone_weights)

    def detect_satellite(self, img_bgr, conf: float = 0.5):
        """
        Детекция на спутниковом фото.
        Возвращает (исходное_изображение, список боксов [x1,y1,x2,y2,cls_id]).
        """
        results = self.satellite_model.predict(img_bgr, conf=conf, verbose=False)
        r = results[0]
        boxes = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            boxes.append([int(x1), int(y1), int(x2), int(y2), cls_id])
        return img_bgr, boxes, r.names

    def detect_drone(self, frame_bgr, conf: float = 0.5):
        """
        Детекция на кадре с дрона.
        Возвращает (исходный кадр, список боксов [x1,y1,x2,y2,cls_id]).
        """
        results = self.drone_model.predict(frame_bgr, conf=conf, verbose=False)
        r = results[0]
        boxes = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            boxes.append([int(x1), int(y1), int(x2), int(y2), cls_id])
        return frame_bgr, boxes, r.names


# ================== УТИЛИТЫ ДЛЯ ИЗОБРАЖЕНИЙ ==================

def cv2_to_tk(img_bgr):
    """Преобразует BGR (cv2) -> PhotoImage (Tkinter)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil)


# ================== ТЕПЛОВАЯ КАРТА ==================

class HeatmapGenerator:
    @staticmethod
    def generate_heatmap(image_shape, boxes, radius=30, class_weights=None):
        """
        Строит серую карту плотности по списку боксов.
        boxes: [x1,y1,x2,y2,cls_id]
        class_weights: опциональный словарь {cls_id: weight}
        """
        h, w = image_shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)

        if class_weights is None:
            # пример: мусор (0) весим сильнее, водоросли (1) чуть слабее, пятно (2) отдельно
            class_weights = {0: 1.0, 1: 0.7, 2: 1.2}

        for x1, y1, x2, y2, cls in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            weight = class_weights.get(cls, 1.0)

            temp = np.zeros((h, w), dtype=np.float32)
            cv2.circle(temp, (cx, cy), radius, weight, -1)
            k = radius * 3 | 1  # радиус*3 и делаем нечётным
            temp = cv2.GaussianBlur(temp, (k, k), radius)

            heat += temp

        if heat.max() > 0:
            heat = heat / heat.max() * 255.0
        return heat.astype(np.uint8)

    @staticmethod
    def overlay_heatmap(base_img_bgr, boxes, alpha=0.6):
        """Накладывает цветную тепловую карту на base_img_bgr."""
        gray = HeatmapGenerator.generate_heatmap(base_img_bgr.shape, boxes)
        heat_color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        mask = gray > 10
        overlay = base_img_bgr.copy()

        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                (1 - alpha) * overlay[:, :, c] + alpha * heat_color[:, :, c],
                overlay[:, :, c],
            )

        return overlay
