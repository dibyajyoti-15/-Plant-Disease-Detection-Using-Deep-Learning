import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout
import os

# ----------------------------
# Dataset paths
# ----------------------------
train_dir = "dataset/train"
val_dir = "dataset/validation"

# ----------------------------
# Image generators
# ----------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

val_gen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ----------------------------
# Improved CNN Model
# ----------------------------
model = Sequential([

    Input(shape=(128,128,3)),

    Conv2D(32, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(512, activation="relu"),
    Dropout(0.5),

    Dense(256, activation="relu"),
    Dropout(0.3),

    Dense(train_data.num_classes, activation="softmax")

])

# ----------------------------
# Compile model
# ----------------------------
model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

# ----------------------------
# Train model
# ----------------------------
history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=20

)

# ----------------------------
# Save model
# ----------------------------
os.makedirs("model", exist_ok=True)

model.save("model/plant_model.h5")

print("Model trained and saved successfully!")