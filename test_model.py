import cv2
from model.predict import predict_emotion

img = cv2.imread(r"C:\strokeemotionmodel\fer2013\test\happy\PrivateTest_2569530.jpg")
if img is None:
    print("❌ Image not found. Check file path.")
    exit()

emotion, conf = predict_emotion(img)

print("Prediction:", emotion)
print("Confidence:", conf)