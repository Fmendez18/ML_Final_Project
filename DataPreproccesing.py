import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Both folders from the grassknoted dataset
TRAIN_PATH = "/Users/santiagolandinez/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train"
IMG_SIZE   = 64

def load_images(folder_path):
    images = []
    labels = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)

        if not os.path.isdir(label_path):
            continue

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# Load both folders
print("Loading training folder...")
X_train_raw, y_train_raw = load_images(TRAIN_PATH)
print(f"Train folder loaded: {X_train_raw.shape}")





