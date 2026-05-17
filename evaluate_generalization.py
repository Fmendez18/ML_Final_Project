import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score,precision_score,recall_score,
    f1_score,confusion_matrix,classification_report,)
from torch.utils.data import DataLoader, TensorDataset
import argparse


IMG_SIZE = 64
BATCH_SIZE = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "asl_cnn.pth"),
    os.path.join(BASE_DIR, "models", "asl_cnn.pth"),
]

DEFAULT_KAGGLE_DATASET = "debashishsau/aslamerican-sign-language-aplhabet-dataset"

# Device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# helper functions
def find_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_images(folder_path, allowed_classes=None, max_per_class=200):
    images = []
    labels = []
    class_counts = {}

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)

        if not os.path.isdir(label_path):
            continue

        if allowed_classes is not None and label not in allowed_classes:
            continue

        class_counts[label] = 0

        for img_file in os.listdir(label_path):
            if class_counts[label] >= max_per_class:
                break

            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img)
            labels.append(label)
            class_counts[label] += 1

    return np.array(images), np.array(labels)


def resolve_image_root(path: str) -> str:
    for root, dirs, files in os.walk(path):
        subdirs = [
            d for d in dirs
            if os.path.isdir(os.path.join(root, d))
        ]
        if len(subdirs) >= 10:
            return root

    return path


class ASL_CNN(nn.Module):
    def __init__(self, num_classes):
        super(ASL_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="External generalization evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the external dataset folder containing class subfolders",
    )
    parser.add_argument(
        "--use_kagglehub",
        action="store_true",
        help="Download the external dataset with kagglehub instead of using --data_dir",
    )
    args = parser.parse_args()

    classes_path = os.path.join(PROCESSED_DATA_DIR, "classes.npy")

    print("BASE_DIR:", BASE_DIR)
    print("PROCESSED_DATA_DIR:", PROCESSED_DATA_DIR)
    print("CLASSES_PATH:", classes_path)

    model_path = find_existing_path(MODEL_CANDIDATES)
    if model_path is None:
        print("Checked model candidates:")
        for candidate in MODEL_CANDIDATES:
            print(" -", candidate)
        raise FileNotFoundError("Could not find asl_cnn.pth in project root or models/")

    print("MODEL_PATH:", model_path)

    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Could not find classes at: {classes_path}")

    classes = np.load(classes_path, allow_pickle=True)
    known_classes = set(classes.tolist())

    if args.use_kagglehub:
        import kagglehub
        raw_path = kagglehub.dataset_download(DEFAULT_KAGGLE_DATASET)
        external_dir = resolve_image_root(raw_path)
        print("Downloaded dataset path:", raw_path)
        print("Resolved image root:", external_dir)
    else:
        if args.data_dir is None:
            raise ValueError("Provide --data_dir or use --use_kagglehub")
        external_dir = args.data_dir

    print("\nLoading external images...")
    X_ext_raw, y_ext_raw = load_images(external_dir, allowed_classes=known_classes, max_per_class=500)

    if len(X_ext_raw) == 0:
        raise ValueError(
            f"No images found in {external_dir}. "
            "Check the folder path and make sure it contains class subfolders."
        )

    print(f"External images: {X_ext_raw.shape}")
    print(f"External labels: {y_ext_raw.shape}")
    print(f"Classes kept: {np.unique(y_ext_raw)}")

    # To encode the labels using the same class order as one of the already trained model
    encoder = LabelEncoder()
    encoder.fit(classes)
    y_ext = encoder.transform(y_ext_raw)

    # PyTorch --> (batch, channels, height, width)
    X_ext = np.transpose(X_ext_raw, (0, 3, 1, 2))
    X_ext_t = torch.tensor(X_ext, dtype=torch.float32)
    y_ext_t = torch.tensor(y_ext, dtype=torch.long)

    test_dataset = TensorDataset(X_ext_t, y_ext_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # building and loading of the model
    num_classes = len(classes)
    model = ASL_CNN(num_classes).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # interface
    all_preds = []
    all_true = []

    print("\nRunning external evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # metrics
    acc = accuracy_score(all_true, all_preds)
    prec_macro = precision_score(all_true, all_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_true, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_true, all_preds, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print("EXTERNAL GENERALIZATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec_macro:.4f} (macro)")
    print(f"Recall:    {rec_macro:.4f} (macro)")
    print(f"F1 score:  {f1_macro:.4f} (macro)")

    cm = confusion_matrix(all_true, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            all_true,
            all_preds,
            target_names=classes.astype(str),
            zero_division=0,
        )
    )

    wrong = np.where(all_preds != all_true)[0]
    print(f"\nMisclassified samples: {len(wrong)} / {len(all_true)}")
    if len(wrong) > 0:
        print("First 10 mistakes:")
        for idx in wrong[:10]:
            true_label = classes[all_true[idx]]
            pred_label = classes[all_preds[idx]]
            print(f"  sample {idx}: true={true_label}, pred={pred_label}")


if __name__ == "__main__":
    main()