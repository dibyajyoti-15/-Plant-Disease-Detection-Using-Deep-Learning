# 🌿 Plant Disease Detection Using Deep Learning

An AI-powered web application that analyzes plant leaf images to
identify selected potato and tomato diseases using Convolutional Neural
Networks (CNNs). The system includes a plant-image validation step to
help reject non-plant images.

## 📌 Project Overview

Plant diseases can reduce crop quality and agricultural productivity.
Identifying diseases early can help farmers take appropriate action.

This project uses deep learning and image processing to classify plant
leaf images. Users can upload an image through a web interface and
receive a predicted disease category, a confidence score, and basic
remedy information for supported classes.

## ✨ Features

-   **Plant Image Validation:** Checks whether an uploaded image is
    likely to contain a plant.
-   **Disease Classification:** Predicts one of six supported potato and
    tomato leaf classes.
-   **Confidence Score:** Displays the model's prediction confidence.
-   **Remedy Information:** Provides basic remedy information for
    supported classes.
-   **Web Interface:** Uses Flask to handle image uploads and display
    prediction results.
-   **Image Preprocessing:** Resizes images to 128 × 128 pixels and
    normalizes pixel values.
-   **Model Training:** Includes scripts for training CNN models and
    preparing datasets.
-   **Testing Utilities:** Includes scripts for testing model
    predictions and checking dependencies.

## 🦠 Supported Classes

The current prediction code supports these six classes:

  Crop     Class
  -------- --------------
  Potato   Healthy
  Potato   Early Blight
  Potato   Late Blight
  Tomato   Healthy
  Tomato   Early Blight
  Tomato   Late Blight

**Note:** Predictions are limited to the classes supported by the
trained model. The system is not a universal plant disease detector.

## 🛠️ Technologies Used

-   Python
-   TensorFlow and Keras
-   Convolutional Neural Networks (CNN)
-   OpenCV
-   NumPy
-   Flask
-   HTML and Jinja2 templates
-   PlantVillage dataset

## 📂 Project Structure

``` text
Plant-Disease-Detection-Using-Deep-Learning/
├── app.py
├── plant_detector.py
├── predict.py
├── train_model.py
├── prepare_plant_images.py
├── split_dataset.py
├── create folders.py
├── test.py
├── test_detector.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── model/
│   ├── plant_model.h5
│   └── plant_detector.h5
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
└── uploads/
```

**Important:** The `templates`, `model`, `dataset`, and `uploads`
directories must be created or generated as required. The two trained
model files and `templates/index.html` were not among the uploaded files
reviewed so far.

## ⚙️ Installation and Setup

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR-USERNAME/Plant-Disease-Detection-Using-Deep-Learning.git
cd Plant-Disease-Detection-Using-Deep-Learning
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

``` bash
python -m venv venv
```

Activate it on Windows:

``` bash
venv\\Scripts\\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the PlantVillage dataset and organize the supported potato and
tomato classes into the directory structure expected by the training
scripts.

Run the dataset preparation script when the source dataset is arranged
correctly:

``` bash
python split_dataset.py
```

The dataset directory names and paths must match those configured in the
scripts.

### 5. Train the models

Train the disease classification model:

``` bash
python train_model.py
```

Train the plant-image validation model:

``` bash
python plant_detector.py
```

The training scripts must be configured with the correct dataset paths.
The resulting model files should be saved under the `model/` directory
using the filenames expected by `predict.py`.

### 6. Run the web application

Once the trained models and HTML template are available, start the
application:

``` bash
python app.py
```

Open the local address displayed in your terminal, normally:

``` text
http://127.0.0.1:5000
```

Upload a plant leaf image to view the prediction.

## 🔍 How It Works

1.  **Image Upload:** The user uploads a leaf image through the Flask
    web application.
2.  **Preprocessing:** OpenCV reads the image, converts its color
    format, resizes it, and normalizes the pixel values.
3.  **Plant Validation:** The first model estimates whether the image
    belongs to a plant.
4.  **Disease Prediction:** If the image passes the plant-validation
    threshold, the second model predicts a supported disease class.
5.  **Confidence Check:** The prediction is checked against a configured
    confidence threshold.
6.  **Result Display:** The application displays the prediction and,
    where available, basic remedy information.

## ⚠️ Limitations

-   Classification is limited to the six classes configured in the
    current prediction code.
-   Accuracy depends on the quality of the training data and the
    similarity between uploaded images and training images.
-   Real-world images with complex backgrounds, poor lighting, or
    unusual symptoms may produce incorrect predictions.
-   A confidence score is not a guarantee of a correct diagnosis.
-   Remedy information is basic and should not replace advice from a
    qualified agricultural expert.

## 🚀 Future Enhancements

-   Support additional crops and disease categories.
-   Evaluate the model on real-world field photographs.
-   Improve detection of non-plant images.
-   Add disease descriptions and more detailed treatment guidance.
-   Deploy the application online and improve mobile usability.

## 🎓 Project Information

-   **Project Title:** Plant Disease Detection Using Deep Learning
-   **Domain:** Artificial Intelligence, Deep Learning, and Computer
    Vision
-   **Degree:** Bachelor of Technology in Computer Science and
    Engineering
-   **Institution:** C. V. Raman Global University, Bhubaneswar, Odisha

## 👥 Project Team

-   Shiv Raj
-   Kundan Sahu
-   Rohit Kumar
-   Dibyajyoti Dalai

## 🙏 Acknowledgements

We acknowledge the PlantVillage dataset and the developers of Python,
TensorFlow, Keras, OpenCV, and Flask for the resources used in this
project.

------------------------------------------------------------------------

**Disclaimer:** This application is an educational project and provides
preliminary image-based predictions, not a definitive agricultural
diagnosis.
