from deepface import DeepFace

# Map to your 4 classes
def map_emotion(e):
    if e == "happy":
        return "happy"
    if e == "sad":
        return "sad"
    if e == "angry":
        return "angry"
    return "neutral"

def predict_emotion(face_img):
    # DeepFace expects RGB
    result = DeepFace.analyze(
        face_img,
        actions=['emotion'],
        enforce_detection=False,
        detector_backend="opencv"  # fast & simple
    )

    # DeepFace returns list of dicts
    emo = result[0]["dominant_emotion"]
    conf = max(result[0]["emotion"].values()) / 100.0

    return map_emotion(emo), conf