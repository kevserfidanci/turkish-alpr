import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from paddleocr import PaddleOCR
import sqlite3
from datetime import datetime
import re

def correct_turkish_plate(ocr_text):
    text = ocr_text.upper()
    province_code_map = {'O':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8'}
    text = re.sub(r'\s+','',text)  # Remove all whitespace
    raw_code = text[:2]
    province_code = ''
    for ch in raw_code:
        if ch.isdigit():
            province_code += ch
        elif ch in province_code_map:
            province_code += province_code_map[ch]
        else:
            province_code += '0'
    remainder = text[2:]
    remainder = re.sub(r'[QWX]','', remainder)  # Remove letters not in Turkish plates
    remainder = re.sub(r'[^A-Z0-9]', '', remainder)  # Remove non-alphanumeric characters
    
    letters = ''
    digits = ''
    i = 0

    while i < len(remainder) and remainder[i].isalpha() and len(letters) < 3:
        letters += remainder[i]
        i += 1
    while i < len(remainder) and remainder[i].isdigit() and len(digits) < 4:
        digits += remainder[i]
        i += 1

    result = province_code
    if letters:
        result += ' ' + letters
    if digits:
        result += ' ' + digits
    return result

def is_plate_valid(plate):
    plate_no_space = plate.replace(" ", "")
    if len(plate_no_space) < 6:
        return False
    pattern = r"^\d{2}[A-Z]{1,3}\d{2,4}$"
    return bool(re.match(pattern, plate_no_space))

class PlateReader(BaseSolution):
    def __init__(self, db_path="plates.db", **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT,
                track_id INTEGER,
                class_name TEXT,
                numberplate TEXT
            )
        ''')
        self.conn.commit()

        self.logged_ids = set()
        self.plate_texts_per_id = {}
        self.required_confirmations = 4

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def enhance_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 70:
            alpha = 3.0
            beta = 40
        elif mean_brightness > 180:
            alpha = 0.7
            beta = -30
        elif mean_brightness < 120:
            alpha = 2.0
            beta = 20
        else:
            alpha = 1.5
            beta = 10

        hist_eq = cv2.equalizeHist(gray)
        enhanced = cv2.convertScaleAbs(hist_eq, alpha=alpha, beta=beta)

        # Gaussian Blur for noise reduction
       # enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return enhanced

    def perform_ocr(self, image_array):
        if image_array is None or not isinstance(image_array, np.ndarray):
            return ""
        enhanced_img = self.enhance_image(image_array)
        results = self.ocr.ocr(enhanced_img)
        raw_text = ' '.join([res[1][0] for res in results[0]] if results and results[0] else "")
        corrected_text = correct_turkish_plate(raw_text)
        return corrected_text

    def process_frame(self, frame):
        self.annotator = Annotator(frame, line_width=self.line_width)
        self.extract_tracks(frame)

        current_time = datetime.now()
        plates_to_show = []
        height, width = frame.shape[:2]

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            x1, y1, x2, y2 = map(int, box)

            pad_x, pad_y = 20, 10
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)

            cropped_image = frame[y1:y2, x1:x2]

            ocr_text = self.perform_ocr(cropped_image).strip()
            if not ocr_text or not is_plate_valid(ocr_text):
                continue

            if track_id in self.logged_ids:
                label = f"ID: {track_id} Plate: {ocr_text} (Saved)"
                self.annotator.box_label(box, label=label, color=colors(track_id, True))
                continue

            if track_id not in self.plate_texts_per_id:
                self.plate_texts_per_id[track_id] = []
            self.plate_texts_per_id[track_id].append(ocr_text)

            if len(self.plate_texts_per_id[track_id]) >= self.required_confirmations:
                best_plate = max(self.plate_texts_per_id[track_id], key=len)
                datetime_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                class_name = self.names[int(cls)]

                self.cursor.execute('''
                    INSERT INTO plates (datetime, track_id, class_name, numberplate)
                    VALUES (?, ?, ?, ?)
                ''', (datetime_str, track_id, class_name, best_plate))
                self.conn.commit()

                self.logged_ids.add(track_id)
                plates_to_show.append((track_id, best_plate, datetime_str))
                self.plate_texts_per_id.pop(track_id)
                label = f"ID: {track_id} Plate: {best_plate} (Saved)"
            else:
                label = f"ID: {track_id} Plate: {ocr_text} (Pending)"

            self.annotator.box_label(box, label=label, color=colors(track_id, True))

        self.display_output(frame)
        return frame, plates_to_show

class App:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("License Plate Recognition")

        self.cap = cv2.VideoCapture(video_path)

        region_points = [(0, 145), (1018, 145)]
        self.plate_reader = PlateReader(
            region=region_points,
            model="y11_e20.pt",
            line_width=2
        )

        self.video_label = tk.Label(root)
        self.video_label.pack(side=tk.LEFT)

        self.tree = ttk.Treeview(root, columns=("id", "plate", "datetime"), show='headings', height=20)
        self.tree.heading("id", text="ID")
        self.tree.heading("plate", text="Plate")
        self.tree.heading("datetime", text="Date & Time")
        self.tree.column("datetime", width=150)
        self.tree.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.detected_plates = []

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        frame = cv2.resize(frame, (1020, 500))

        processed_frame, plates = self.plate_reader.process_frame(frame)

        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.detected_plates.extend(plates)
        if len(self.detected_plates) > 100:
            self.detected_plates = self.detected_plates[-100:]

        self.update_treeview()
        self.root.after(30, self.update_frame)

    def update_treeview(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for pid, plate, datetime_str in self.detected_plates:
            self.tree.insert("", "end", values=(pid, plate, datetime_str))

if __name__ == "__main__":
    root = tk.Tk()
    video_path = "ornek.mp4"  
    app = App(root, video_path)
    root.mainloop()
