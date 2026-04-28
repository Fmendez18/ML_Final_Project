import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Both folders from the grassknoted dataset
TRAIN_PATH = "/Users/fmendez/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train"
IMG_SIZE   = 64

def load_images(folder_path, excluded_classes=None):
    images = []
    labels = []

    for label in os.listdir(folder_path):

        if excluded_classes and label in excluded_classes: #added this to remove J and Z from dataset
            continue

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
X_train_raw, y_train_raw = load_images(
    TRAIN_PATH,
    excluded_classes=["J", "Z"]
)
print("Remaining classes:", np.unique(y_train_raw))
print("Shape without J and Z:", X_train_raw.shape)

#Now we will start checking the preprocessing.
print("STEP 2: Checking class balance...")
print("=" * 50)
unique, counts = np.unique(y_train_raw, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count} images")

min_count = counts.min()
max_count = counts.max()
print(f"\nMin images in a class: {min_count}")
print(f"Max images in a class: {max_count}")

if max_count / min_count > 2:
    print("WARNING: Dataset is imbalanced! Consider balancing classes!!!")
else:
    print("Dataset is balanced! Lets gooooo")




print("\n" + "=" * 50)
print("STEP 3: Splitting Data (Train / Validation / Test)")
print("=" * 50)

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_train_raw,
    y_train_raw,
    test_size=0.3,
    stratify=y_train_raw, #Stratify data as a best practice
    random_state=2026
)

print(f"Training set shape: {X_train.shape}")
print(f"Temporary set shape: {X_temp.shape}")

# Second split: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=2026
)

print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

print("\n" + "=" * 50)
print("STEP 4: Encoding Labels (fit on train only)")
print("=" * 50)

encoder = LabelEncoder()

# Fit ONLY on training labels
y_train_encoded = encoder.fit_transform(y_train)

# Transform validation and test using fitted encoder
y_val_encoded = encoder.transform(y_val)
y_test_encoded = encoder.transform(y_test)

print("Classes found:", encoder.classes_)
print(f"Number of classes: {len(encoder.classes_)}")

print("\nSample encoding:")
for cls in encoder.classes_[:5]:
    print(f"{cls} → {encoder.transform([cls])[0]}") #This helps us check if the encoding is correct

print("\nEncoded label shapes:")
print("y_train_encoded:", y_train_encoded.shape)
print("y_val_encoded:", y_val_encoded.shape)
print("y_test_encoded:", y_test_encoded.shape)


print("\n" + "=" * 50)
print("STEP 5: Saving processed data...")
print("=" * 50)

SAVE_DIR = "processed_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save feature arrays
np.save(f"{SAVE_DIR}/X_train.npy", X_train)
np.save(f"{SAVE_DIR}/X_val.npy", X_val)
np.save(f"{SAVE_DIR}/X_test.npy", X_test)

# Save encoded labels
np.save(f"{SAVE_DIR}/y_train.npy", y_train_encoded)
np.save(f"{SAVE_DIR}/y_val.npy", y_val_encoded)
np.save(f"{SAVE_DIR}/y_test.npy", y_test_encoded)

# Save class mapping
np.save(f"{SAVE_DIR}/classes.npy", encoder.classes_)

print(f"X_train saved: {X_train.shape}")
print(f"y_train saved: {y_train_encoded.shape}")
print(f"X_val saved:   {X_val.shape}")
print(f"y_val saved:   {y_val_encoded.shape}")
print(f"X_test saved:  {X_test.shape}")
print(f"y_test saved:  {y_test_encoded.shape}")
print(f"Classes saved: {encoder.classes_}")

print("\nPreprocessing complete! Ready for training.")