import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, url_for
from predict import predict_disease


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


DISEASE_INFO = {

    "Potato_Early_Blight": {
        "name": "Potato Early Blight",
        "remedy": "Use Mancozeb fungicide and remove infected leaves."
    },

    "Potato_Late_Blight": {
        "name": "Potato Late Blight",
        "remedy": "Apply Copper-based fungicide and avoid excess moisture."
    },

    "Potato_Healthy": {
        "name": "Healthy Potato Plant",
        "remedy": "No disease detected. The plant appears healthy."
    },

    "Tomato_Early_Blight": {
        "name": "Tomato Early Blight",
        "remedy": "Spray Chlorothalonil fungicide and remove infected leaves."
    },

    "Tomato_Late_Blight": {
        "name": "Tomato Late Blight",
        "remedy": "Use Metalaxyl fungicide and improve drainage."
    },

    "Tomato_Healthy": {
        "name": "Healthy Tomato Plant",
        "remedy": "No disease detected. The plant appears healthy."
    }
}


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None
    remedy = None
    is_plant = None
    plant_probability = None
    image_url = None

    if request.method == "POST":

        file = request.files.get("file")

        if file and file.filename != "":

            filepath = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(filepath)

            # Run the two-stage prediction
            result, confidence, is_plant = predict_disease(filepath)

            # Show uploaded image
            image_url = url_for(
                "uploaded_file",
                filename=file.filename
            )

            # -------------------------
            # NON-PLANT IMAGE
            # -------------------------
            if not is_plant:

                plant_probability = confidence

                result = "Invalid Image - Please upload a plant image"

                confidence = None

                remedy = None

            # -------------------------
            # PLANT IMAGE
            # -------------------------
            else:

                disease_info = DISEASE_INFO.get(result)

                if disease_info:

                    plant_probability = 1.0

                    remedy = disease_info["remedy"]

                    result = disease_info["name"]

                else:

                    remedy = None

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        remedy=remedy,
        is_plant=is_plant,
        plant_probability=plant_probability,
        image_url=image_url
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):

    from flask import send_from_directory

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


if __name__ == "__main__":

    app.run(debug=True)