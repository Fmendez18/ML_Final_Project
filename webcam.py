import cv2
import numpy as np
import torch
import torch.nn as nn
import pyautogui
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_DIR     = "models"
DATA_DIR      = "processed_data"
IMG_SIZE      = 64
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE    = 0.85      # minimum confidence to accept a prediction
COOLDOWN      = 3       # seconds between typing each letter
BOX_SIZE      = 350       # size of the hand capture box

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
class ASL_CNN(nn.Module):
    def __init__(self, num_classes):
        super(ASL_CNN, self)._init_()
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


classes     = np.load(f"{DATA_DIR}/classes.npy")
num_classes = len(classes)

model = ASL_CNN(num_classes).to(DEVICE)
model.load_state_dict(torch.load(f"{MODEL_DIR}/asl_cnn_best.pth",
                                  map_location=DEVICE))
model.eval()
print("Model loaded!")
print(f"Classes: {classes}")

# ── HELPER: PREPROCESS FRAME ──────────────────────────────────────────────────
def preprocess(roi):
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))           # HWC → CHW
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0).to(DEVICE)            # add batch dimension
    return img

# ── HELPER: PREDICT ───────────────────────────────────────────────────────────
def predict(roi):
    tensor = preprocess(roi)
    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    return classes[predicted.item()], confidence.item()

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
cap           = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()
last_typed    = time.time()
current_word  = ""

print("\n🤟 Sign Language Translator Running!")
print("Place your hand inside the GREEN box")
print("Press 'Q' to quit | Press 'SPACE' to add space | Press 'BACKSPACE' to delete\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, retrying...")
        time.sleep(0.1)
        continue

    frame = cv2.flip(frame, 1)                   # mirror the webcam
    h, w  = frame.shape[:2]

    # Define the hand capture box in the center-right of the frame
    x1 = w // 2 - BOX_SIZE // 2
    y1 = h // 2 - BOX_SIZE // 2
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE

    # Extract the region of interest (hand area)
    roi = frame[y1:y2, x1:x2]

    # Run prediction
    letter, confidence = predict(roi)

    # Only type if confidence is high enough and cooldown has passed
    now = time.time()
    if confidence >= CONFIDENCE and (now - last_typed) >= COOLDOWN:
        if letter == "space":
            pyautogui.press("space")
            current_word += " "
        elif letter == "del":
            pyautogui.press("backspace")
            current_word = current_word[:-1]
        elif letter == "nothing":
            pass                                 # do nothing for "nothing" class
        else:
            pyautogui.typewrite(letter.lower())  # types into whatever is focused
            current_word += letter
        last_typed = now
        print(f"Typed: {letter} (confidence: {confidence:.2f})")

    # ── DRAW UI ───────────────────────────────────────────────────────────────
    # Draw the capture box
    box_color = (0, 255, 0) if confidence >= CONFIDENCE else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Show prediction and confidence
    cv2.putText(frame, f"Letter: {letter}",
                (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # Show current word being built
    cv2.putText(frame, f"Word: {current_word}",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show cooldown bar
    elapsed  = min(now - last_typed, COOLDOWN)
    bar_w    = int((elapsed / COOLDOWN) * BOX_SIZE)
    cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_w, y2 + 15), (0, 255, 0), -1)
    cv2.putText(frame, "Ready" if bar_w == BOX_SIZE else "Wait...",
                (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("ASL Sign Language Translator", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        pyautogui.press("space")
        current_word += " "
    elif key == 8:                               # backspace key
        pyautogui.press("backspace")
        current_word = current_word[:-1]

cap.release()
cv2.destroyAllWindows()
print(f"\nSession ended. Full text typed: {current_word}")
