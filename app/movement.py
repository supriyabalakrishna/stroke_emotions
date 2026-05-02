import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "movement_model.pkl")

movement_model = None
loaded = False  # ensure we try loading only once

def _load_model_once():
    global movement_model, loaded
    if loaded:
        return
    loaded = True
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                movement_model = pickle.load(f)
        else:
            print("⚠️ movement_model.pkl not found, using fallback")
    except Exception as e:
        print("⚠️ Failed to load movement model:", e)

def detect_movement(features):
    _load_model_once()

    if movement_model is None:
        return "Not Available"

    features = np.array(features).reshape(1, -1)
    prediction = movement_model.predict(features)[0]
    return str(prediction)