import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


IMG_SIZE = 150
MODEL_PATH = "output/cnn_model.keras"
TEST_DIR = "./test"
OUTPUT_DIR = "output/gradcam"
LAST_CONV_LAYER_NAME = "last_conv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


CLASS_NAMES = ["Pneumonia", "Normal"]



model = load_model(MODEL_PATH)
_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32))
print("Model loaded.")
print([layer.name for layer in model.layers])


def load_image(img_path, img_size=150):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img_array = img.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)   # (150,150,1)
    img_array = np.expand_dims(img_array, axis=0)    # (1,150,150,1)
    return img, img_array



def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    last_conv_layer = model.get_layer(last_conv_layer_name)

    last_conv_layer_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    last_conv_index = None
    for i, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            last_conv_index = i
            break

    for layer in model.layers[last_conv_index + 1:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        last_conv_output = last_conv_layer_model(img_tensor)
        tape.watch(last_conv_output)

        preds = classifier_model(last_conv_output)

        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)

    if grads is None:
        raise ValueError("Gradient is None. Check whether last_conv layer is correct.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = tf.reduce_sum(last_conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy(), preds.numpy()



def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if len(original_img.shape) == 2:
        original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        original_img_color = original_img

    superimposed_img = cv2.addWeighted(original_img_color, 1 - alpha, heatmap_color, alpha, 0)
    return heatmap_color, superimposed_img


def run_gradcam_on_image(img_path, true_label=None, save_prefix="sample"):
    original_img, img_array = load_image(img_path, IMG_SIZE)

    heatmap, preds = make_gradcam_heatmap(
        img_array=img_array,
        model=model,
        last_conv_layer_name=LAST_CONV_LAYER_NAME
    )

    pred_label = int((preds[0][0] > 0.5))
    pred_prob = float(preds[0][0])

    heatmap_color, overlay_img = overlay_heatmap(original_img, heatmap)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap="gray")
    title = "Original"
    if true_label is not None:
        title += f"\nTrue: {CLASS_NAMES[true_label]}"
    plt.title(title)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Pred: {CLASS_NAMES[pred_label]}\nScore: {pred_prob:.3f}")
    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{save_prefix}_gradcam.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    print(f"Prediction score: {pred_prob:.4f}")
    print(f"Predicted label: {CLASS_NAMES[pred_label]}")

    return pred_label, pred_prob


sample_img_path = os.path.join(TEST_DIR, "NORMAL", os.listdir(os.path.join(TEST_DIR, "NORMAL"))[0])

run_gradcam_on_image(sample_img_path, true_label=1, save_prefix="test_normal_example")