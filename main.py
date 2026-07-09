import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import os
import cv2
import numpy as np

from utils import GeoUtils, cv2_to_tk, draw_boxes, overlay_heatmap
from drone import DroneController

try:
    from models import ModelManager
    HAS_MODELS = os.path.exists("weights/satellite_best.pt") and os.path.exists("weights/drone_best.pt")
except Exception:
    HAS_MODELS = False


def fake_satellite_boxes(width, height):
    # заглушка на случай, если веса моделей не подложены в weights/
    boxes = []
    boxes.append([int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.65), 2])
    boxes.append([int(width * 0.28), int(height * 0.30), int(width * 0.42), int(height * 0.45), 1])
    boxes.append([int(width * 0.55), int(height * 0.28), int(width * 0.70), int(height * 0.42), 1])
    boxes.append([int(width * 0.45), int(height * 0.48), int(width * 0.58), int(height * 0.60), 0])
    boxes.append([int(width * 0.32), int(height * 0.50), int(width * 0.44), int(height * 0.62), 0])
    return boxes


def fake_underwater_boxes(width, height):
    boxes = []
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
    boxes.append([int(width * 0.10), int(height * 0.15), int(width * 0.28), int(height * 0.35), 1])
    boxes.append([int(width * 0.65), int(height * 0.55), int(width * 0.82), int(height * 0.75), 1])
    return boxes


class MirrikoApp:
    """
    1) Загрузить фото мусорного пятна
    2) Загрузить видео с дрона
    3) Запустить миссию - YOLO анализирует фото и кадр (или заглушка, если весов нет)
    4) Построить общую тепловую карту
    """

    COLORS = {
        "bg": "#0f172a",
        "panel": "#1e293b",
        "text": "#e5e7eb",
        "accent": "#38bdf8",
        "success": "#10b981",
        "warning": "#f59e0b",
        "danger": "#ef4444",
    }

    def __init__(self, root):
        self.root = root
        title = "Миррико - тепловая карта мусорного пятна"
        title += " (реальные модели)" if HAS_MODELS else " (демо-режим, веса не найдены)"
        self.root.title(title)
        self.root.geometry("1440x830")
        self.root.configure(bg=self.COLORS["bg"])

        self.photo_bgr = None
        self.video_path = None
        self.last_video_frame = None
        self.video_running = False

        self.sat_boxes = []
        self.uw_boxes = []

        self.models = ModelManager() if HAS_MODELS else None

        self.raft_lat, self.raft_lon = 55.7558, 37.6173
        self.target_lat = self.raft_lat + 0.009
        self.target_lon = self.raft_lon + 0.009

        self.drone = DroneController(self.raft_lat, self.raft_lon)
        self.drone.on_position_update = self.update_coords
        self.drone.on_arrival = self.on_drone_arrived

        self._tk_refs = {}

        self.build_ui()
        self.update_coords()
        self.update_telemetry()
        def build_ui(self):
        C = self.COLORS

        top = tk.Frame(self.root, bg=C["panel"], padx=10, pady=10)
        top.pack(fill="x", padx=10, pady=(10, 0))

        self.lbl_status = tk.Label(top, text="Шаг 1: загрузите фото мусорного пятна",
                                    bg=C["panel"], fg=C["accent"], font=("Arial", 13, "bold"))
        self.lbl_status.grid(row=0, column=0, sticky="w", padx=10)

        self.lbl_coords_drone = tk.Label(top, text="АНПА: -", bg=C["panel"], fg=C["text"], font=("Courier", 11))
        self.lbl_coords_drone.grid(row=0, column=1, padx=20)

        self.lbl_coords_target = tk.Label(top, text=f"Цель: {self.target_lat:.4f}, {self.target_lon:.4f}",
                                           bg=C["panel"], fg=C["text"], font=("Courier", 11))
        self.lbl_coords_target.grid(row=0, column=2, padx=20)

        self.lbl_dist_depth = tk.Label(top, text="Дистанция: - м | Глубина: 0.0 м",
                                        bg=C["panel"], fg=C["warning"], font=("Arial", 11))
        self.lbl_dist_depth.grid(row=0, column=3, padx=20)

        center = tk.Frame(self.root, bg=C["bg"])
        center.pack(fill="both", expand=True, padx=10, pady=8)

        left = tk.LabelFrame(center, text="Видео с дрона (трансляция)", bg=C["panel"], fg=C["accent"],
                              font=("Arial", 12, "bold"))
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.video_label = tk.Label(left, bg="black")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

        self.lbl_video_info = tk.Label(self.video_label, text="Видео не загружено", bg="black", fg="lime",
                                        font=("Courier", 10))
        self.lbl_video_info.place(x=10, y=10)

        right = tk.Frame(center, bg=C["bg"], width=420)
        right.pack(side="right", fill="both", expand=False, padx=(6, 0))
        right.pack_propagate(False)

        def make_img_panel(parent, title, attr):
            frame = tk.LabelFrame(parent, text=title, bg=C["panel"], fg=C["text"], font=("Arial", 11))
            frame.pack(fill="both", expand=True, pady=4)
            lbl = tk.Label(frame, bg="black", fg="white", text="-", width=42, height=9)
            lbl.pack(fill="both", expand=True, padx=4, pady=4)
            setattr(self, attr, lbl)

        make_img_panel(right, "Спутник -> YOLO_1", "sat_label")
        make_img_panel(right, "Дрон -> YOLO_2", "uw_label")
        make_img_panel(right, "Тепловая карта мусора+водорослей", "heat_label")

        bottom = tk.Frame(self.root, bg=C["panel"], padx=10, pady=10)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        def btn(parent, text, cmd, state="normal", bg=None, fg=None):
            b = tk.Button(parent, text=text, command=cmd, state=state, font=("Arial", 10, "bold"),
                          bg=bg or "#334155", fg=fg or "white", relief="flat", padx=10, pady=6)
            b.pack(side="left", padx=5)
            return b

        self.btn_load_photo = btn(bottom, "1. Загрузить фото (спутник)", self.load_photo)
        self.btn_load_video = btn(bottom, "2. Загрузить видео с дрона", self.load_video)
        self.btn_start = btn(bottom, "3. Запустить миссию", self.start_mission, state="disabled",
                              bg="#10b981", fg="white")
        self.btn_heatmap = btn(bottom, "4. Построить тепловую карту", self.make_heatmap, state="disabled")

        self.lbl_verdict = tk.Label(bottom, text="РЕЗУЛЬТАТ: ожидание", bg=C["panel"], fg="gray",
                                     font=("Arial", 15, "bold"))
        self.lbl_verdict.pack(side="right", padx=20)

    def load_photo(self):
        path = filedialog.askopenfilename(title="Выберите фото мусорного пятна",
                                           filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Ошибка", "Не удалось прочитать файл.")
            return

        self.photo_bgr = cv2.resize(img, (400, 300))
        self._show(self.sat_label, self.photo_bgr, "_tk_photo_orig")
        self.lbl_status.config(text="Фото загружено. Шаг 2: загрузите видео с дрона.", fg=self.COLORS["accent"])
        self._check_ready()

    def load_video(self):
        path = filedialog.askopenfilename(title="Выберите видео с дрона",
                                           filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov")])
        if not path:
            return

        self.video_path = path
        self.lbl_video_info.config(text="Видео загружено, воспроизведение...")
        self._check_ready()

        if not self.video_running:
            self.video_running = True
            threading.Thread(target=self._video_loop, daemon=True).start()

    def _check_ready(self):
        if self.photo_bgr is not None and self.video_path is not None:
            self.btn_start.config(state="normal")
            self.lbl_status.config(text="Готово. Шаг 3: запустите миссию.", fg=self.COLORS["success"])

    def _video_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.lbl_video_info.config(text="Ошибка: не удалось открыть видео")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 1 / 25

        while self.video_running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (640, 480))
            self.last_video_frame = frame.copy()

            cv2.circle(frame, (30, 28), 9, (0, 0, 255), -1)
            cv2.putText(frame, "DRONE CAM REC", (48, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            tk_img = cv2_to_tk(frame)
            try:
                self.video_label.configure(image=tk_img)
                self.video_label.image = tk_img
            except tk.TclError:
                break

            time.sleep(delay)
        cap.release()

    def start_mission(self):
        if self.photo_bgr is None or self.video_path is None:
            messagebox.showwarning("Внимание", "Сначала загрузите фото и видео.")
            return

        self.btn_start.config(state="disabled")
        self.btn_load_photo.config(state="disabled")
        self.btn_load_video.config(state="disabled")
        self.lbl_status.config(text="Миссия: YOLO_1 анализирует фото, АНПА плывёт к цели...",
                                fg=self.COLORS["accent"])

        h, w = self.photo_bgr.shape[:2]
        if self.models:
            self.sat_boxes = self.models.detect_satellite(self.photo_bgr)
        else:
            self.sat_boxes = fake_satellite_boxes(w, h)

        sat_vis = draw_boxes(self.photo_bgr, self.sat_boxes)
        self._show(self.sat_label, sat_vis, "_tk_sat_vis")

        self.drone.set_target(self.target_lat, self.target_lon)
        self.drone.start_mission()

    def on_drone_arrived(self):
        def _cb():
            self.lbl_status.config(text="АНПА прибыл. YOLO_2 анализирует последний кадр...",
                                    fg=self.COLORS["warning"])

            base_frame = self.last_video_frame if self.last_video_frame is not None else np.zeros((300, 400, 3), dtype=np.uint8)
            base_frame = cv2.resize(base_frame, (400, 300))

            h, w = base_frame.shape[:2]
            if self.models:
                self.uw_boxes = self.models.detect_drone(base_frame)
            else:
                self.uw_boxes = fake_underwater_boxes(w, h)

            uw_vis = draw_boxes(base_frame, self.uw_boxes)
            self._show(self.uw_label, uw_vis, "_tk_uw_vis")

            self.lbl_status.config(text="Обе модели отработали. Шаг 4: постройте тепловую карту.",
                                    fg=self.COLORS["success"])
            self.btn_heatmap.config(state="normal")
            self.btn_load_photo.config(state="normal")
            self.btn_load_video.config(state="normal")

        self.root.after(0, _cb)

    def make_heatmap(self):
        if self.photo_bgr is None:
            messagebox.showwarning("Внимание", "Нет исходного фото.")
            return
        if not self.sat_boxes and not self.uw_boxes:
            messagebox.showwarning("Внимание", "Нет детекций. Сначала запустите миссию.")
            return

        combined = self.sat_boxes + self.uw_boxes
        heat_img = overlay_heatmap(self.photo_bgr, combined, alpha=0.6)
        self._show(self.heat_label, heat_img, "_tk_heat")

        self.lbl_status.config(text="Тепловая карта построена. Красные зоны - высокая концентрация мусора.",
                                fg=self.COLORS["success"])
        self.lbl_verdict.config(text="РЕЗУЛЬТАТ: мусорное пятно подтверждено", fg=self.COLORS["success"])

    def _show(self, label, img_bgr, ref_key):
        tk_img = cv2_to_tk(img_bgr)
        label.configure(image=tk_img, text="")
        label.image = tk_img
        self._tk_refs[ref_key] = tk_img

    def update_coords(self):
        def _cb():
            self.lbl_coords_drone.config(text=f"АНПА: {self.drone.lat:.4f}, {self.drone.lon:.4f}")
            dist = GeoUtils.calculate_distance(self.drone.lat, self.drone.lon, self.target_lat, self.target_lon)
            self.lbl_dist_depth.config(text=f"Дистанция: {int(dist)} м | Глубина: {self.drone.depth:.1f} м")
        self.root.after(0, _cb)

    def update_telemetry(self):
        if self.drone.is_moving:
            rssi = -40 - np.random.randint(0, 15)
            self.lbl_video_info.config(text=f"RSSI: {rssi} dBm | FPS: 30 | Связь: поплавок")
        self.root.after(1000, self.update_telemetry)

    def on_close(self):
        self.video_running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MirrikoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
