import math
import numpy as np
import cv2
from PIL import Image, ImageTk


class GeoUtils:
    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        R = 6371e3
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class MockDetections:
    """
    Имитация двух YOLO-моделей.
    class_id: 0 — мусор, 1 — водоросли, 2 — крупное пятно.
    """

    @staticmethod
    def satellite_boxes(width, height):
        """Искусственные детекции для спутникового фото."""
        boxes = []
        # Крупное мусорное пятно по центру (модель пятен)
        boxes.append([int(width * 0.25), int(height * 0.25),
                      int(width * 0.75), int(height * 0.65), 2])
        # Водоросли
        boxes.append([int(width * 0.28), int(height * 0.30),
                      int(width * 0.42), int(height * 0.45), 1])
        boxes.append([int(width * 0.55), int(height * 0.28),
                      int(width * 0.70), int(height * 0.42), 1])
        # Мусор
        boxes.append([int(width * 0.45), int(height * 0.48),
                      int(width * 0.58), int(height * 0.60), 0])
        boxes.append([int(width * 0.32), int(height * 0.50),
                      int(width * 0.44), int(height * 0.62), 0])
        return boxes

    @staticmethod
    def underwater_boxes(width, height):
        """Искусственные детекции для подводного кадра с дрона."""
        boxes = []
        # Плотный кластер мусора в центре
        for _ in range(8):
            cx = int(width * 0.45 + np.random.randn() * width * 0.06)
            cy = int(height * 0.45 + np.random.randn() * height * 0.06)
            w = int(width * 0.07)
            h = int(height * 0.06)
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(width - 1, cx + w // 2)
            y2 = min(height - 1, cy + h // 2)
            boxes.append([x1, y1, x2, y2, 0])
        # Водоросли по краям
        boxes.append([int(width * 0.10), int(height * 0.15),
                      int(width * 0.28), int(height * 0.35), 1])
        boxes.append([int(width * 0.65), int(height * 0.55),
                      int(width * 0.82), int(height * 0.75), 1])
        return boxes


class ImageUtils:
    @staticmethod
    def cv2_to_tk(img_bgr):
        """BGR (cv2) → PhotoImage (Tkinter)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        return ImageTk.PhotoImage(pil)

    @staticmethod
    def draw_boxes(img_bgr, boxes, names=None):
        """
        Рисует bounding box'ы на изображении.
        class_colors: 0 — зелёный (мусор), 1 — циановый (водоросли), 2 — красный (пятно).
        """
        colors = {0: (0, 255, 0), 1: (0, 200, 255), 2: (0, 0, 255)}
        default_names = {0: "Мусор", 1: "Водоросли", 2: "Пятно"}
        label_map = names if names else default_names

        vis = img_bgr.copy()
        for x1, y1, x2, y2, cls in boxes:
            color = colors.get(cls, (255, 255, 255))
            label = label_map.get(cls, str(cls))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis, label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA,
            )
        return vis


class HeatmapGenerator:
    @staticmethod
    def generate_heatmap(image_shape, boxes, radius=30, class_weights=None):
        """
        Строит серую карту плотности по списку боксов.
        Чем больше боксов пересекается в области — тем «горячее» она на карте.
        """
        h, w = image_shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)

        if class_weights is None:
            class_weights = {0: 1.0, 1: 0.7, 2: 1.2}

        for x1, y1, x2, y2, cls in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            weight = class_weights.get(cls, 1.0)

            temp = np.zeros((h, w), dtype=np.float32)
            cv2.circle(temp, (cx, cy), radius, weight, -1)
            k = radius * 3 | 1  # нечётный размер ядра
            temp = cv2.GaussianBlur(temp, (k, k), radius)
            heat += temp

        if heat.max() > 0:
            heat = heat / heat.max() * 255.0
        return heat.astype(np.uint8)

    @staticmethod
    def overlay_heatmap(base_img_bgr, boxes, alpha=0.6):
        """
        Накладывает цветную тепловую карту (COLORMAP_JET) на base_img_bgr.
        Синий — низкая плотность, красный — высокая.
        """
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
