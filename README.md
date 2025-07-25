# License Plate Recognition for Turkish Plates with YOLOv11 and PaddleOCR

This project performs license plate detection and recognition from video frames using a custom-trained YOLOv11 model combined with PaddleOCR for character recognition. It is specifically designed to work with Turkish license plates and includes post-processing to correct OCR errors based on the Turkish license plate format.

---

## Features

- Vehicle license plate detection using a custom YOLOv11n model trained on a dedicated license plate dataset
- OCR on detected plates with PaddleOCR
- Post-processing and correction of Turkish license plate text to improve accuracy
- Tracking plates across frames and saving confirmed plate readings to a SQLite database with timestamps and track IDs
- Real-time video display with bounding boxes and plate labels using Tkinter GUI
- Display of detected plates with timestamps in a table

---

## Project Scope

This project is specifically designed for recognizing Turkish license plates. The OCR output is corrected according to the Turkish license plate system rules to improve recognition accuracy.

---

## Video Example

The example video used for testing is a segment from this YouTube video:

[License Plate Recognition Demo](https://www.youtube.com/watch?v=lk1ASAvAcqQ)

---

## Dataset

The license plate detection model was trained on the following dataset:

[Roboflow Universe License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)

---

## Model Training Details

- Model: YOLOv11n (YOLOv11 Nano variant)
- Training epochs: 20
- Batch size: 16
- Dataset: License Plate Recognition Dataset (link above)
- Custom weights file used: `y11_e20.pt`

---

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- Pillow (`Pillow`)
- Ultralytics YOLO library with YOLOv11 support
- PaddleOCR (`paddleocr`)
- SQLite3 (standard with Python)
- Tkinter (usually included with Python)

---

## Notes

- The OCR model uses PaddleOCR with English language configuration; you can customize it if needed.  
- Turkish license plate correction is applied to improve OCR results.  
- Detected plates are saved in `plates.db` SQLite database file with timestamp, track ID, class name, and plate text.  
- The GUI displays video feed on the left and a table with detected plates on the right.  
- Make sure your environment supports Tkinter GUI.  
- The YOLOv11 model weights (`y11_e20.pt`) should be present in the project folder or provide the correct path.

- 

## How to Run

1. Clone this repository
2. Place your video file (e.g. `ornek.mp4`) in the project directory or adjust the path accordingly in the main script.
3. Make sure you have the YOLOv11 weights file `y11_e20.pt` in the project directory.
4. Install dependencies, for example:

```bash
pip install opencv-python pillow ultralytics paddleocr

