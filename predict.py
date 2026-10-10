import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load both models
disease_model = load_model("model/plant_model.h5")
plant_detector = load_model("model/plant_detector.h5")

# Disease classes
classes = [
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight",
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight"
]


def predict_disease(image_path):

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (128, 128))

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # ==========================================
    # STEP 1: PLANT / NON-PLANT
    # ==========================================

    plant_prediction = float(
        plant_detector.predict(img, verbose=0)[0][0]
    )

    print("Plant probability:", f"{plant_prediction:.2%}")

    PLANT_THRESHOLD = 0.70

    if plant_prediction < PLANT_THRESHOLD:
        return (
            "Invalid Image - Please upload a plant image",
            plant_prediction,
            False
        )

    # ==========================================
    # STEP 2: DISEASE PREDICTION
    # ==========================================

    prediction = disease_model.predict(img, verbose=0)[0]

    index = np.argmax(prediction)

    confidence = float(prediction[index])

    disease = classes[index]

    # ==========================================
    # STEP 3: DISEASE CONFIDENCE
    # ==========================================

    DISEASE_THRESHOLD = 0.60

    if confidence < DISEASE_THRESHOLD:
        return (
            "Unable to confidently identify the disease",
            confidence,
            True
        )

    return disease, confidence, True


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    image_path = r"dataset\test\Potato_Early_Blight\002a55fb-7a3d-4a3a-aca8-ce2d5ebc6925___RS_Early.B 8170.JPG"

    result, confidence, is_plant = predict_disease(image_path)

    print()
    print("==========================================")
    print("PREDICTION RESULT")
    print("==========================================")

    print("Result     :", result)
    print("Confidence :", f"{confidence:.2%}")