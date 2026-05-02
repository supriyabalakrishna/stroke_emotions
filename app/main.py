from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from app.emotion import detect_emotion
from app.movement import detect_movement

app = FastAPI()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    pad = 20
    y1 = max(0, y - pad)
    y2 = min(frame.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(frame.shape[1], x + w + pad)

    return frame[y1:y2, x1:x2]

@app.get("/")
def home():
    return {"message": "Stroke Recovery API Running"}

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face = detect_face(frame)

    if face is None:
        return {"error": "No face detected"}

    emotion, confidence = detect_emotion(face)

    # dummy movement input (replace later)
    movement = detect_movement([0.1, 0.2, 0.3, 0.4])

    return {
        "emotion": emotion,
        "confidence": round(confidence, 2),
        "movement": movement
    }