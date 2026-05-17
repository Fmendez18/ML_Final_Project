import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from itertools import product

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = "processed_data"
MODEL_DIR  = "models"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


PARAM_GRID = {
    "learning_rate": [0.001, 0.0001],
    "batch_size":    [32, 64],
    "epochs":        [5, 10]
}


print("\n" + "=" * 50)
print("Loading data...")
print("=" * 50)

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
classes = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_val   = np.transpose(X_val,   (0, 3, 1, 2))

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

num_classes = len(classes)
print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")


class ASL_CNN(nn.Module):
    def __init__(self, num_classes):
        super(ASL_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x



def train_and_evaluate(lr, batch_size, epochs):
    model     = ASL_CNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    # Training
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_val_t), batch_size):
            batch    = X_val_t[i:i+batch_size].to(DEVICE)
            _, preds = model(batch).max(1)
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(y_val, np.array(all_preds))
    return acc, model



print("\n" + "=" * 50)
print("Starting Grid Search...")
print(f"Total combinations: "
      f"{len(PARAM_GRID['learning_rate']) * len(PARAM_GRID['batch_size']) * len(PARAM_GRID['epochs'])}")
print("=" * 50)

results      = []
best_acc     = 0.0
best_params  = {}
best_model   = None
combo_num    = 0

for lr, bs, ep in product(PARAM_GRID["learning_rate"],
                           PARAM_GRID["batch_size"],
                           PARAM_GRID["epochs"]):
    combo_num += 1
    print(f"\nCombo {combo_num}: lr={lr} | batch_size={bs} | epochs={ep}")

    acc, model = train_and_evaluate(lr, bs, ep)
    results.append({"lr": lr, "batch_size": bs, "epochs": ep, "val_acc": acc})

    print(f"  → Val Accuracy: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc    = acc
        best_params = {"learning_rate": lr, "batch_size": bs, "epochs": ep}
        best_model  = model


print("\n" + "=" * 50)
print("GRID SEARCH RESULTS")
print("=" * 50)
print(f"{'LR':<10} {'Batch':<10} {'Epochs':<10} {'Val Acc':<10}")
print("-" * 40)
for r in sorted(results, key=lambda x: x["val_acc"], reverse=True):
    print(f"{r['lr']:<10} {r['batch_size']:<10} {r['epochs']:<10} {r['val_acc']*100:.2f}%")

print("\n" + "=" * 50)
print("BEST HYPERPARAMETERS FOUND")
print("=" * 50)
print(f"Learning Rate: {best_params['learning_rate']}")
print(f"Batch Size:    {best_params['batch_size']}")
print(f"Epochs:        {best_params['epochs']}")
print(f"Val Accuracy:  {best_acc*100:.2f}%")


os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(best_model.state_dict(),
           os.path.join(MODEL_DIR, "asl_cnn_best_tuned.pth"))
print(f"\nBest model saved to {MODEL_DIR}/asl_cnn_best_tuned.pth")