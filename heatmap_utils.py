import numpy as np
import cv2
from PIL import Image, ImageTk


def cv2_to_tk(img_bgr):
    """Преобразует cv2-изображение (BGR) в PhotoImage для Tkinter."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil)


def generate_fake_boxes(width, height, num_boxes=10):
    """
    Генерируем ИСКУССТВЕННЫЕ bounding-box'ы как будто от YOLO.
    Формат: [x1, y1, x2, y2].
    """
    boxes = []
    for _ in range(num_boxes):
        w = np.random.randint(width // 10, width // 5)
        h = np.random.randint(height // 10, height // 5)
        x1 = np.random.randint(0, width - w)
        y1 = np.random.randint(0, height - h)
        x2 = x1 + w
        y2 = y1 + h
        boxes.append([x1, y1, x2, y2])
    return boxes


def generate_heatmap_from_boxes(image_shape, boxes, radius=30):
    """
    Строим серую тепловую карту по списку боксов.
    Идея: центр каждого бокса «подогревает» окрестность Гауссом.
    """
    h, w = image_shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        temp = np.zeros((h, w), dtype=np.float32)
        cv2.circle(temp, (cx, cy), radius, 1.0, -1)
        k = radius * 3 | 1  # делаем размер ядра нечетным
        temp = cv2.GaussianBlur(temp, (k, k), radius)

        heat += temp

    if heat.max() > 0:
        heat = heat / heat.max() * 255.0

    return heat.astype(np.uint8)


def overlay_heatmap(base_img_bgr, boxes, alpha=0.6):
    """
    Накладывает тепловую карту поверх base_img_bgr.
    base_img_bgr — это фото мусорного пятна (сверху).
    boxes — список боксов, полученных (искусственно) с видео.
    """
    gray = generate_heatmap_from_boxes(base_img_bgr.shape, boxes)
    heat_color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # область, где тепловая карта ненулевая
    mask = gray > 10

    overlay = base_img_bgr.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask,
            (1 - alpha) * overlay[:, :, c] + alpha * heat_color[:, :, c],
            overlay[:, :, c],
        )

    return overlay
