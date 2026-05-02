from model.predict import predict_emotion

def get_emotion(face_img):
    emotion, confidence = predict_emotion(face_img)
    return {
        "emotion": emotion,
        "confidence": confidence
    }