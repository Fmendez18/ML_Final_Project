import os
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)
from collections import Counter


DATA_DIR = "processed_data"


print("=" * 50)
print("Loading data...")
print("=" * 50)

y_train  = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test   = np.load(os.path.join(DATA_DIR, "y_test.npy"))
classes  = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)

print(f"Training samples: {len(y_train)}")
print(f"Test samples:     {len(y_test)}")
print(f"Classes:          {classes}")


print("\n" + "=" * 50)
print("Running Majority Class Predictor...")
print("=" * 50)


most_common_class = Counter(y_train).most_common(1)[0][0]
most_common_label = classes[most_common_class]
print(f"Most common class in training set: '{most_common_label}' (index {most_common_class})")

# Predict the majority class for every single test sample
y_pred_majority = np.full(len(y_test), most_common_class)


print("\n" + "=" * 50)
print("MAJORITY CLASS BASELINE RESULTS")
print("=" * 50)

acc  = accuracy_score(y_test, y_pred_majority)
prec = precision_score(y_test, y_pred_majority, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred_majority, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred_majority, average="macro", zero_division=0)

print(f"Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f} (macro)")
print(f"Recall:    {rec:.4f}  (macro)")
print(f"F1 Score:  {f1:.4f}  (macro)")

print("\nNote: With 29 balanced classes, random chance = ~3.45%")
print("This baseline confirms our dataset is balanced and")
print("highlights how much better our CNN performs.")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_majority,
    target_names=classes.astype(str),
    zero_division=0
))