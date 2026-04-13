import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop

from sklearn.metrics import classification_report, confusion_matrix


img_size = 150
data_dir = "."
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")

LR_LIST = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]   

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
                img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
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

# reshape
x_train = x_train.reshape(-1, img_size, img_size, 1)
x_test = x_test.reshape(-1, img_size, img_size, 1)
x_val = x_val.reshape(-1, img_size, img_size, 1)


datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train)


def build_model(lr):
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(150,150,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D())

    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D())

    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D())

    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D())

    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


for LR in LR_LIST:

    print(f"\n========== Training with LR = {LR} ==========")

    model = build_model(LR)

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=2,
        factor=0.3,
        min_lr=1e-6
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=12,
        validation_data=(x_val, y_val),
        callbacks=[lr_scheduler],
        verbose=1
    )


    epochs = range(1, 13)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, history.history['accuracy'], label='Train Acc')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Acc')
    plt.title(f"Accuracy (LR={LR})")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.title(f"Loss (LR={LR})")
    plt.legend()

    plt.savefig(f"output/training_lr_{LR}.png")
    plt.close()


    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

    report = classification_report(y_test, y_pred, target_names=["Pneumonia", "Normal"])
    cm = confusion_matrix(y_test, y_pred)

    print(report)
    print(cm)


    with open(f"output/cnn_results_lr_{LR}.txt", "w") as f:
        f.write(report)
        f.write("\n")
        f.write(np.array2string(cm))


    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Pneumonia", "Pred Normal"],
                yticklabels=["True Pneumonia", "True Normal"])
    plt.title(f"CNN Confusion Matrix (LR={LR})")
    plt.savefig(f"output/cnn_cm_lr_{LR}.png")
    plt.close()