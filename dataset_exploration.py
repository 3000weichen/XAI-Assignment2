import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = "."
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")

def get_image_files(folder):
    valid_ext = (".jpeg", ".jpg", ".png", ".bmp")
    return [
        f for f in os.listdir(folder)
        if not f.startswith(".") and f.lower().endswith(valid_ext)
    ]
    
def collect_labels(folder, split_name):
    data = []
    
    normal_files = get_image_files(os.path.join(folder, "NORMAL"))
    pneumonia_files = get_image_files(os.path.join(folder, "PNEUMONIA"))
    
    for _ in normal_files:
        data.append({"Split": split_name, "Label": "Normal"})
    
    for _ in pneumonia_files:
        data.append({"Split": split_name, "Label": "Pneumonia"})
    
    return data

all_data = []
all_data.extend(collect_labels(train_dir, "Train"))
all_data.extend(collect_labels(val_dir, "Validation"))
all_data.extend(collect_labels(test_dir, "Test"))

df = pd.DataFrame(all_data)

os.makedirs("output", exist_ok=True)

sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))

ax = sns.countplot(data=df, x="Split", hue="Label")

plt.title("Class Distribution in Train, Validation, and Test Sets", fontsize=14)
plt.xlabel("Dataset Split", fontsize=12)
plt.ylabel("Count", fontsize=12)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.savefig("output/data_distribution_all_sets.png", dpi=300, bbox_inches="tight")
plt.close()
