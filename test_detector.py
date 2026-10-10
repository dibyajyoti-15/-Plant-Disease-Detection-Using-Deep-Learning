import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/plant_detector.h5")

image_path = r"dataset\test\Potato_Early_Blight\002a55fb-7a3d-4a3a-aca8-ce2d5ebc6925___RS_Early.B 8170.JPG"

img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

prediction = float(model.predict(img, verbose=0)[0][0])

print()
print("==========================================")
print("PLANT DETECTOR TEST")
print("==========================================")
print("Plant probability    :", f"{prediction:.2%}")
print("Non-plant probability:", f"{1 - prediction:.2%}")

if prediction >= 0.70:
    print("Prediction           : PLANT")
else:
    print("Prediction           : NON-PLANT")