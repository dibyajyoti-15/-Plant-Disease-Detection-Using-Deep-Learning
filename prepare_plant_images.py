import os
import shutil
import random

# ==========================================
# SETTINGS
# ==========================================

SOURCE_DIR = "dataset/train"
VAL_SOURCE_DIR = "dataset/validation"

PLANT_TRAIN_DIR = "dataset/plant_detector/train/plant"
PLANT_VAL_DIR = "dataset/plant_detector/validation/plant"

TRAIN_LIMIT = 2000
VAL_LIMIT = 400

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")

# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs(PLANT_TRAIN_DIR, exist_ok=True)
os.makedirs(PLANT_VAL_DIR, exist_ok=True)

# ==========================================
# GET IMAGES FROM EXISTING DATASET
# ==========================================

def get_images(folder):

    images = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.endswith(IMAGE_EXTENSIONS):

                images.append(os.path.join(root, file))

    return images


print("Finding existing plant images...")

train_images = get_images(SOURCE_DIR)
val_images = get_images(VAL_SOURCE_DIR)

print("Plant images found in train dataset     :", len(train_images))
print("Plant images found in validation dataset:", len(val_images))

# ==========================================
# SHUFFLE
# ==========================================

random.shuffle(train_images)
random.shuffle(val_images)

# ==========================================
# LIMIT NUMBER OF IMAGES
# ==========================================

train_images = train_images[:TRAIN_LIMIT]
val_images = val_images[:VAL_LIMIT]

# ==========================================
# COPY TRAINING PLANT IMAGES
# ==========================================

print()
print("Copying training plant images...")

for i, source in enumerate(train_images):

    destination = os.path.join(
        PLANT_TRAIN_DIR,
        f"plant_{i}.jpg"
    )

    shutil.copy2(source, destination)

# ==========================================
# COPY VALIDATION PLANT IMAGES
# ==========================================

print("Copying validation plant images...")

for i, source in enumerate(val_images):

    destination = os.path.join(
        PLANT_VAL_DIR,
        f"plant_{i}.jpg"
    )

    shutil.copy2(source, destination)

# ==========================================
# RESULT
# ==========================================

print()
print("==========================================")
print("PLANT DATASET CREATED")
print("==========================================")

print("Training plant images   :", len(train_images))
print("Validation plant images :", len(val_images))

print()
print("Train folder:")
print(PLANT_TRAIN_DIR)

print()
print("Validation folder:")
print(PLANT_VAL_DIR)

print()
print("Done!")