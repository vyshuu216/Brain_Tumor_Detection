import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

IMG_SIZE = 224


def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image


def predict_tumor(image_path):

    image = preprocess_image(image_path)

    prediction = model.predict(image, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        label = "Tumor"
        confidence = probability * 100
    else:
        label = "No Tumor"
        confidence = (1 - probability) * 100

    return {
        "Prediction": label,
        "Confidence": round(confidence,2)
    }


if __name__ == "__main__":

    path = input("Enter MRI Image Path: ")

    result = predict_tumor(path)

    print(result)