import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


IMG_SIZE = 150
MODEL_PATH = "output/cnn_model.keras"
TEST_DIR = "./test"
OUTPUT_DIR = "output/gradcam_misclassified"
LAST_CONV_LAYER_NAME = "last_conv"
NUM_SAMPLES = 8   

os.makedirs(OUTPUT_DIR, exist_ok=True)


CLASS_NAMES = ["Pneumonia", "Normal"]


model = load_model(MODEL_PATH)
_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32))
print("Model loaded.")
print("Model layers:")
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
        raise ValueError("Gradient is None. Please check last_conv layer name.")

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


def predict_image(img_path):
    _, img_array = load_image(img_path, IMG_SIZE)
    pred_prob = model.predict(img_array, verbose=0)[0][0]

    pred_label = 1 if pred_prob > 0.5 else 0

    return pred_label, pred_prob


def collect_test_images(test_dir):
    image_info = []

    for label_name, true_label in [("PNEUMONIA", 0), ("NORMAL", 1)]:
        folder = os.path.join(test_dir, label_name)

        for img_name in os.listdir(folder):
            if img_name.startswith("."):
                continue
            img_path = os.path.join(folder, img_name)
            image_info.append((img_path, true_label))

    return image_info


def find_misclassified_samples(test_dir):
    all_images = collect_test_images(test_dir)
    misclassified = []

    for img_path, true_label in all_images:
        pred_label, pred_prob = predict_image(img_path)

        if pred_label != true_label:
            misclassified.append({
                "img_path": img_path,
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_prob": pred_prob
            })

    return misclassified


def save_gradcam_result(sample_info, index):
    img_path = sample_info["img_path"]
    true_label = sample_info["true_label"]
    pred_label = sample_info["pred_label"]
    pred_prob = sample_info["pred_prob"]

    original_img, img_array = load_image(img_path, IMG_SIZE)

    heatmap, _ = make_gradcam_heatmap(
        img_array=img_array,
        model=model,
        last_conv_layer_name=LAST_CONV_LAYER_NAME
    )

    _, overlay_img = overlay_heatmap(original_img, heatmap)

    filename = os.path.basename(img_path)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap="gray")
    plt.title(f"Original\nTrue: {CLASS_NAMES[true_label]}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.title(
        f"Pred: {CLASS_NAMES[pred_label]}\nScore: {pred_prob:.3f}"
    )
    plt.axis("off")

    plt.tight_layout()

    save_name = f"{index+1:02d}_{filename}_true_{CLASS_NAMES[true_label]}_pred_{CLASS_NAMES[pred_label]}.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path



misclassified_samples = find_misclassified_samples(TEST_DIR)

print(f"\nTotal misclassified samples found: {len(misclassified_samples)}")

if len(misclassified_samples) == 0:
    print("No misclassified samples found.")
else:
    selected_samples = misclassified_samples[:NUM_SAMPLES]

    summary_path = os.path.join(OUTPUT_DIR, "misclassified_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total misclassified samples found: {len(misclassified_samples)}\n")
        f.write(f"Saved first {len(selected_samples)} samples.\n\n")

        for i, sample in enumerate(selected_samples):
            save_path = save_gradcam_result(sample, i)

            line = (
                f"{i+1}. File: {os.path.basename(sample['img_path'])}\n"
                f"   True label: {CLASS_NAMES[sample['true_label']]}\n"
                f"   Pred label: {CLASS_NAMES[sample['pred_label']]}\n"
                f"   Score: {sample['pred_prob']:.4f}\n"
                f"   Saved to: {save_path}\n\n"
            )
            f.write(line)
            print(line)

    print(f"Summary saved to: {summary_path}")