import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

img_size = 150
data_dir = "."   
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")

os.makedirs("output", exist_ok=True)

def load_data(folder):
    data = []
    labels = []

    for label_name in ["PNEUMONIA", "NORMAL"]:
        path = os.path.join(folder, label_name)
        class_num = 0 if label_name == "PNEUMONIA" else 1

        for img_name in os.listdir(path):
            if img_name.startswith("."):
                continue
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                data.append(img)
                labels.append(class_num)
            except:
                continue

    return np.array(data), np.array(labels)

print("Loading data...")

x_train, y_train = load_data(train_dir)
x_test, y_test = load_data(test_dir)
x_val, y_val = load_data(val_dir)

x_train = x_train / 255.0
x_test = x_test / 255.0
x_val = x_val / 255.0

x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)
x_val = x_val.reshape(len(x_val), -1)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

train_report = classification_report(y_train, y_pred_train, target_names=["Pneumonia", "Normal"])
test_report = classification_report(y_test, y_pred_test, target_names=["Pneumonia", "Normal"])

train_cm = confusion_matrix(y_train, y_pred_train)
test_cm = confusion_matrix(y_test, y_pred_test)

print("\nClassification Report - Train:")
print(train_report)
print("\nConfusion Matrix - Train:")
print(train_cm)

print("\nClassification Report - Test:")
print(test_report)
print("\nConfusion Matrix - Test:")
print(test_cm)

with open("output/logreg_results.txt", "w") as f:
    f.write("Classification Report - Train:\n")
    f.write(train_report)
    f.write("\nConfusion Matrix - Train:\n")
    f.write(np.array2string(train_cm))
    f.write("\n\nClassification Report - Test:\n")
    f.write(test_report)
    f.write("\nConfusion Matrix - Test:\n")
    f.write(np.array2string(test_cm))

matrix_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=["Pneumonia", "Normal"]
)
matrix_display.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("output/logreg_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()