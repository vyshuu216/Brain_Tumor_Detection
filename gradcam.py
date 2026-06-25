import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

IMG_SIZE = 224


def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image


def generate_gradcam(image_path):

    img = preprocess_image(image_path)

    base_model = model.layers[0]

    last_conv_layer = base_model.get_layer("out_relu")

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, x = grad_model(img)

        tape.watch(conv_outputs)

        x = model.layers[1](x)
        x = model.layers[2](x)
        x = model.layers[3](x)
        x = model.layers[4](x)
        predictions = model.layers[5](x)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()

    original = cv2.imread(image_path)

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    superimposed = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    # Create static folder if not present
    os.makedirs("static", exist_ok=True)

    filename = os.path.basename(image_path)

    name = os.path.splitext(filename)[0]

    output_path = f"static/uploads/{name}_gradcam.jpg"

    cv2.imwrite(
        output_path,
        cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
    )

    # Optional preview when running this file directly
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.title("Original MRI")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(superimposed)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    plt.close()

    return output_path


if __name__ == "__main__":

    image_path = input("Enter MRI Image Path: ").strip()

    saved_path = generate_gradcam(image_path)

    print(f"\nGrad-CAM saved at: {saved_path}")