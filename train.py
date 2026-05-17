import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = "processed_data"
MODEL_DIR   = "models"
IMG_SIZE    = 64
BATCH_SIZE  = 64
EPOCHS      = 10
LEARNING_RATE = 0.0001
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── STEP 1: LOAD PREPROCESSED DATA ───────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 1: Loading preprocessed data...")
print("=" * 50)

X_train = np.load(f"{DATA_DIR}/X_train.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")
X_val   = np.load(f"{DATA_DIR}/X_val.npy")
y_val   = np.load(f"{DATA_DIR}/y_val.npy")

print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")

# ── STEP 2: PREPARE TENSORS ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Preparing PyTorch tensors...")
print("=" * 50)

# PyTorch expects (batch, channels, height, width) not (batch, height, width, channels)
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_val   = np.transpose(X_val,   (0, 3, 1, 2))

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# ── STEP 3: BUILD CNN MODEL ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Building CNN model...")
print("=" * 50)

num_classes = len(np.load(f"{DATA_DIR}/classes.npy"))
print(f"Number of classes: {num_classes}")

class ASL_CNN(nn.Module):
    def __init__(self, num_classes):
        super(ASL_CNN, self).__init__()

        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 channels (RGB)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # 64x64 → 32x32
        )

        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # 32x32 → 16x16
        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # 16x16 → 8x8
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),                              # prevents overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


model = ASL_CNN(num_classes).to(DEVICE)
print(model)

# ── STEP 4: DEFINE LOSS AND OPTIMIZER ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ── STEP 5: TRAINING LOOP ─────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Training...")
print("=" * 50)

train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []

for epoch in range(EPOCHS):

    # ── TRAIN ──────────────────────────────────────────
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()           # clear gradients
        outputs = model(images)         # forward pass
        loss    = criterion(outputs, labels)  # compute loss
        loss.backward()                 # backward pass
        optimizer.step()                # update weights

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100. * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ── VALIDATE ───────────────────────────────────────
    model.eval()
    val_loss    = 0.0
    correct     = 0
    total       = 0

    with torch.no_grad():               # no gradient needed for validation
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total       += labels.size(0)
            correct     += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc  = 100. * correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ── STEP 6: SAVE MODEL AND METRICS ───────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Saving model and metrics...")
print("=" * 50)

os.makedirs(MODEL_DIR, exist_ok=True)

# Save model
torch.save(model.state_dict(), f"{MODEL_DIR}/asl_cnn.pth")
print(f"Model saved to {MODEL_DIR}/asl_cnn.pth")

# Save metrics for evaluate.py
np.save(f"{DATA_DIR}/train_losses.npy", np.array(train_losses))
np.save(f"{DATA_DIR}/val_losses.npy",   np.array(val_losses))
np.save(f"{DATA_DIR}/train_accs.npy",   np.array(train_accs))
np.save(f"{DATA_DIR}/val_accs.npy",     np.array(val_accs))
print("Training metrics saved!")

print(f"\nFinal Train Accuracy: {train_accs[-1]:.2f}%")
print(f"Final Val Accuracy:   {val_accs[-1]:.2f}%")
print("\nTraining complete! Ready for evaluation.")