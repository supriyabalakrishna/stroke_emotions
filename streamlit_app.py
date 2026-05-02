import streamlit as st
import cv2
import time
from collections import Counter

from model.predict import predict_emotion

st.title("😊 Emotion Detection")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if st.button("Start 5s Capture"):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not working")
        st.stop()

    st.write("Capturing...")

    start_time = time.time()
    predictions = []

    frame_window = st.empty()

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_window.image(frame, channels="BGR")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]

            # ensure decent face size
            if w > 100 and h > 100:
                face = frame[y:y+h, x:x+w]

                emotion, conf = predict_emotion(face)

                # only keep decent confidence
                if conf > 0.4:
                    predictions.append(emotion)

    cap.release()

    # FINAL RESULT
    if len(predictions) > 0:
        final = Counter(predictions).most_common(1)[0][0]
    else:
        final = "neutral"

    st.success(f"Final Emotion: {final.upper()}")
    st.write("All predictions:", predictions)