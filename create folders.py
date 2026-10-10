import os

# Main dataset directory
base_dir = "dataset"

# Folder names
splits = ["train", "validation", "test"]

classes = [
    "Potato_Early_Blight",
    "Potato_Late_Blight",
    "Potato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Healthy"
]

# Create folders
for split in splits:
    for cls in classes:
        path = os.path.join(base_dir, split, cls)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

print("\\nAll folders created successfully!")