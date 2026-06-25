from flask import Flask, render_template, request
import os

from predict import predict_tumor
from gradcam import generate_gradcam

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image."

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Prediction
    result = predict_tumor(filepath)

    # Generate Grad-CAM
    gradcam_path = generate_gradcam(filepath)

    return render_template(
        "index.html",
        prediction=result["Prediction"],
        confidence=result["Confidence"],
        image_path=filepath,
        gradcam_path=gradcam_path
    )


if __name__ == "__main__":
    app.run(debug=True)