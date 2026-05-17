import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)


DATA_DIR      = "processed_data"
MODEL_DIR     = "models"
EPOCHS        = 10          
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


print("\n" + "=" * 50)
print("STEP 1: Loading preprocessed data...")
print("=" * 50)

X_train  = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train  = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test   = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test   = np.load(os.path.join(DATA_DIR, "y_test.npy"))
classes  = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")


print("\n" + "=" * 50)
print("STEP 2: Preparing tensors...")
print("=" * 50)

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test  = np.transpose(X_test,  (0, 3, 1, 2))

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True)

print("\n" + "=" * 50)
print("STEP 3: Building Simple 1-Block CNN baseline...")
print("=" * 50)

num_classes = len(classes)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                  
        )

        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc(x)
        return x


model     = SimpleCNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)


print("\n" + "=" * 50)
print("STEP 4: Training Simple CNN baseline...")
print("=" * 50)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
          f"Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}%")


print("\n" + "=" * 50)
print("STEP 5: Evaluating Simple CNN baseline...")
print("=" * 50)

model.eval()
all_preds = []

with torch.no_grad():
    for i in range(0, len(X_test_t), BATCH_SIZE):
        batch    = X_test_t[i:i+BATCH_SIZE].to(DEVICE)
        outputs  = model(batch)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())

all_preds = np.array(all_preds)

acc  = accuracy_score(y_test, all_preds)
prec = precision_score(y_test, all_preds, average="macro", zero_division=0)
rec  = recall_score(y_test, all_preds, average="macro", zero_division=0)
f1   = f1_score(y_test, all_preds, average="macro", zero_division=0)

print("\n" + "=" * 50)
print("SIMPLE CNN BASELINE RESULTS")
print("=" * 50)
print(f"Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f} (macro)")
print(f"Recall:    {rec:.4f}  (macro)")
print(f"F1 Score:  {f1:.4f}  (macro)")

print("\nClassification Report:")
print(classification_report(
    y_test, all_preds,
    target_names=classes.astype(str),
    zero_division=0
))


os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "simple_cnn_baseline.pth"))
print(f"\nBaseline model saved to {MODEL_DIR}/simple_cnn_baseline.pth")