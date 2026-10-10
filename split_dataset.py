import os
import random
import shutil

# Path of the extracted PlantVillage dataset
source_dir = "PlantVillage"

# Destination dataset folder
target_dir = "dataset"

# Mapping original folder names to our project folder names
classes = {
    "Potato___Early_blight": "Potato_Early_Blight",
    "Potato___Late_blight": "Potato_Late_Blight",
    "Potato___healthy": "Potato_Healthy",
    "Tomato_Early_blight": "Tomato_Early_Blight",
    "Tomato_Late_blight": "Tomato_Late_Blight",
    "Tomato_healthy": "Tomato_Healthy"
}

# Create destination folders automatically
for split in ["train", "validation", "test"]:
    for folder in classes.values():
        os.makedirs(os.path.join(target_dir, split, folder), exist_ok=True)

# Split ratio
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for src_name, dst_name in classes.items():

    source_path = os.path.join(source_dir, src_name)

    images = os.listdir(source_path)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    # Copy training images
    for img in train_imgs:
        shutil.copy(
            os.path.join(source_path, img),
            os.path.join(target_dir, "train", dst_name, img)
        )

    # Copy validation images
    for img in val_imgs:
        shutil.copy(
            os.path.join(source_path, img),
            os.path.join(target_dir, "validation", dst_name, img)
        )

    # Copy testing images
    for img in test_imgs:
        shutil.copy(
            os.path.join(source_path, img),
            os.path.join(target_dir, "test", dst_name, img)
        )

    print(f"{dst_name} completed")

print("\\nDataset split completed successfully!")