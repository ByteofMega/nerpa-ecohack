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


def cv2_to_tk(img_bgr):
    # tkinter не понимает BGR/cv2, конвертим в PhotoImage
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil)


def draw_boxes(img_bgr, boxes, names=None):
    # 0 - мусор, 1 - водоросли, 2 - крупное пятно
    colors = {0: (0, 255, 0), 1: (0, 200, 255), 2: (0, 0, 255)}
    default_names = {0: "Мусор", 1: "Водоросли", 2: "Пятно"}
    label_map = names if names else default_names

    vis = img_bgr.copy()
    for x1, y1, x2, y2, cls in boxes:
        color = colors.get(cls, (255, 255, 255))
        label = label_map.get(cls, str(cls))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def generate_heatmap(image_shape, boxes, radius=30, class_weights=None):
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
        k = radius * 3 | 1  # ядро должно быть нечетным
        temp = cv2.GaussianBlur(temp, (k, k), radius)
        heat += temp

    if heat.max() > 0:
        heat = heat / heat.max() * 255.0
    return heat.astype(np.uint8)


def overlay_heatmap(base_img_bgr, boxes, alpha=0.6):
    gray = generate_heatmap(base_img_bgr.shape, boxes)
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
