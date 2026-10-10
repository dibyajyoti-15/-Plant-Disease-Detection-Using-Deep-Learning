import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout
)
import os

# ==========================================
# DATASET PATHS
# ==========================================

train_dir = "dataset/plant_detector_new/train"
val_dir = "dataset/plant_detector_new/validation"


# ==========================================
# IMAGE DATA GENERATORS
# ==========================================

train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15
)

val_gen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ==========================================
# LOAD DATASET
# ==========================================

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

print()
print("Class mapping:", train_data.class_indices)


# ==========================================
# CNN MODEL
# ==========================================

model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid")
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# TRAIN
# ==========================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

model.save("model/plant_detector.h5")

print()
print("==========================================")
print("PLANT DETECTOR TRAINED SUCCESSFULLY!")
print("==========================================")
print("Model saved at: model/plant_detector.h5")