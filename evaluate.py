import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# setup
DATA_DIR = "processed_data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_cnn.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# STEP 1: LOAD TEST DATA
print("\nLoading test data...")

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
classes = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Classes: {classes}")

# (batch, channels, height, width)
X_test = np.transpose(X_test, (0, 3, 1, 2))

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# STEP 2: RECREATE THE MODEL ARCHITECTURE
class ASL_CNN(nn.Module):
    def __init__(self, num_classes):
        super(ASL_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

num_classes = len(classes)
model = ASL_CNN(num_classes).to(DEVICE)

# STEP 3: LOAD SAVED WEIGHTS
print("\nLoading model weights...")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# STEP 4: RUN INFERENCE
print("\nRunning predictions on test set...")
with torch.no_grad():
    outputs = model(X_test_t.to(DEVICE))
    probabilities = torch.softmax(outputs, dim=1)
    y_pred = torch.argmax(probabilities, dim=1).cpu().numpy()

# STEP 5: COMPUTE METRICS
acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec_macro:.4f} (macro)")
print(f"Recall:    {rec_macro:.4f} (macro)")
print(f"F1 score:  {f1_macro:.4f} (macro)")

# STEP 6: CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# class-wise report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes], zero_division=0))

# STEP 7: SHOW A FEW MISTAKES
wrong = np.where(y_pred != y_test)[0]
print(f"\nMisclassified samples: {len(wrong)} / {len(y_test)}")
if len(wrong) > 0:
    print("First 10 mistakes:")
    for idx in wrong[:10]:
        true_label = classes[y_test[idx]]
        pred_label = classes[y_pred[idx]]
        print(f"  sample {idx}: true={true_label}, pred={pred_label}")