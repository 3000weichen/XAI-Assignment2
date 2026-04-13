import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model("output/cnn_model.keras")

img_size = 150

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.reshape(img, (1, img_size, img_size, 1))
    return img

def compute_saliency(img_array):
    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:, 0]   # binary classification

    grads = tape.gradient(loss, img_tensor)

    saliency = tf.abs(grads)

    # squeeze
    saliency = saliency.numpy()[0, :, :, 0]

    return saliency

def save_saliency(img_path, save_name):
    img = load_image(img_path)
    saliency = compute_saliency(img)

    original = img[0, :, :, 0]

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # saliency
    plt.subplot(1,2,2)
    plt.imshow(saliency, cmap='hot')
    plt.title("Saliency Map")
    plt.axis('off')

    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/{save_name}.png", bbox_inches='tight')
    plt.close()

sample_img = "./test/NORMAL/IM-0001-0001.jpeg"
save_saliency(sample_img, "saliency_example")