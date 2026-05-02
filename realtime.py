import cv2
from collections import defaultdict
from model.predict import predict_emotion

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

emotion_scores = defaultdict(float)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        if w < 100 or h < 100:
            continue

        pad = 20
        y1 = max(0, y-pad)
        y2 = min(frame.shape[0], y+h+pad)
        x1 = max(0, x-pad)
        x2 = min(frame.shape[1], x+w+pad)

        face = frame[y1:y2, x1:x2]

        emotion, conf = predict_emotion(face)

        # 🔥 accumulate weighted score
        if conf > 0.3:
            emotion_scores[emotion] += conf

        label = f"{emotion} ({conf:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(1)

    # 🔥 PRESS 'e' TO GET FINAL RESULT
    if key & 0xFF == ord('e'):
        break

    if key & 0xFF == ord('q'):
        emotion_scores.clear()

cap.release()
cv2.destroyAllWindows()

# ======================
# FINAL OUTPUT
# ======================
if len(emotion_scores) > 0:
    final_emotion = max(emotion_scores, key=emotion_scores.get)
else:
    final_emotion = "neutral"

print("\n✅ FINAL EMOTION:", final_emotion.upper())