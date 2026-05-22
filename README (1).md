# ML_Final_Project — ASL Alphabet Recognition

A CNN-based pipeline for recognising static American Sign Language (ASL) fingerspelling from images, plus a real-time webcam demo. Final group project for the **Machine Learning Foundations** course at IE University (Prof. Matteo Turilli, BDBA 2025).

> The full methodology, results, and reflection are in `report.pdf`. This README covers the code only — how to run it, where things live, and how to reproduce our results.

---

## Results

| Stage | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | 3.70% | 0.26% |
| Simple 1-block CNN | 76.75% | 76.57% |
| **Tuned 3-block CNN — internal test** | **99.85%** | **99.85%** |
| Tuned 3-block CNN — external dataset | 77.89% | 79.54% |

The 22-point gap between internal and external accuracy is our central finding — see the report for the analysis.

---

## Repository layout

The project is provided in **two equivalent forms** — pick whichever you prefer:

```
ML_Final_Project/
├── README.md
├── requirements.txt
│
├── ML_Final_Workflow.ipynb       # Form A: single Colab notebook (recommended for grading)
│
├── DataPreprocessing.py          # Form B: modular scripts, one stage per file
├── majority_baseline.py
├── cnn_baseline.py
├── hyperparameter_tuning.py
├── train.py
├── evaluate.py
├── evaluate_generalization.py
│
├── webcam.py                     # Standalone local demo (cannot run in Colab — see below)
└── collect_data.py               # Utility for capturing custom webcam samples
```

Both forms produce the same results. The notebook chains everything together for easy review; the `.py` files split the same logic into stages for cleaner code organisation.

---

## How to run — Form A: Colab notebook (recommended)

The fastest path for a reviewer.

1. **Open `ML_Final_Workflow.ipynb` in Google Colab.**
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**.

The notebook will:
- Install missing dependencies (`kagglehub`, `opencv-python`)
- Download the ASL Alphabet dataset from Kaggle automatically
- Run preprocessing, baselines, hyperparameter tuning, full training, internal test evaluation, and external-dataset generalisation testing
- Save all artefacts to `/content/ML_Final_Project/` (local to the Colab session)

**Total runtime:** ~30 minutes end-to-end on a T4 GPU.

**Kaggle credentials.** On first run, you'll be prompted to enter Kaggle API credentials. Get them from [kaggle.com](https://www.kaggle.com/) → Settings → API → "Create New Token". One-time setup, ~30 seconds.

**Note on the webcam.** The notebook intentionally does **not** include the webcam demo — Colab runs on a remote server with no camera or GUI access. See the "Webcam demo" section below for how to run it locally.

---

## How to run — Form B: individual Python scripts

If you'd rather run things locally or step-by-step, the `.py` files do the same work as the notebook cells:

```bash
git clone https://github.com/Fmendez18/ML_Final_Project.git
cd ML_Final_Project
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then run the scripts in this order:

| Step | Script | Purpose | Outputs |
|---|---|---|---|
| 1 | `DataPreprocessing.py` | Downloads dataset from Kaggle, resizes, normalises, stratified 70/15/15 split | `processed_data/*.npy` |
| 2 | `majority_baseline.py` | Trivial baseline (~3.70%, the chance floor) | console |
| 3 | `cnn_baseline.py` | Simple 1-block CNN baseline | `models/simple_cnn_baseline.pth` |
| 4 | `hyperparameter_tuning.py` | Grid search over learning rate × batch size × epochs | `models/asl_cnn_best_tuned.pth`, `models/tuning_results.json` |
| 5 | `train.py` | Trains the full 3-block CNN with best hyperparameters | `models/asl_cnn.pth`, `models/asl_cnn_best.pth` |
| 6 | `evaluate.py` | Internal test-set evaluation with confusion matrix | `processed_data/y_pred.npy`, `y_probs.npy` |
| 7 | `evaluate_generalization.py` | External-dataset evaluation (distribution shift) | `processed_data/external_y_pred.npy` |

Each script is independent and re-runnable. Intermediate artefacts persist to disk so a single failed step doesn't lose previous work.

**Best hyperparameters found:** `lr=0.0001`, `batch_size=64`, `epochs=10` → 99.87% validation accuracy.

**GPU is optional but strongly recommended.** On CPU, training takes ~2 hours; on a T4 GPU, ~5 minutes.

---

## Webcam demo (local only)

`webcam.py` runs the trained model on a live webcam feed and types recognised letters into whatever text editor has keyboard focus. **It cannot run in Google Colab** — Colab is a remote server with no camera, no GUI, and no way to type into your local applications. Run it on your own machine.

### Setup

You need three things in the same folder as `webcam.py`:

```
your-folder/
├── webcam.py
├── models/
│   └── asl_cnn_best.pth         ← trained model from train.py / notebook
└── processed_data/
    └── classes.npy              ← label list from DataPreprocessing.py / notebook
```

If you ran the Colab notebook, download `asl_cnn_best.pth` and `classes.npy` from `/content/ML_Final_Project/` first (left sidebar in Colab → Files → right-click → Download).

Install the local-only dependencies:

```bash
pip install opencv-python pyautogui torch numpy
```

### Running

```bash
python webcam.py
```

A window opens showing your webcam feed with a green capture box in the centre. Place your hand in the box and sign letters — the script will type them into whichever window has keyboard focus.

**Important:** click into a text editor (Notes, TextEdit, Notepad, VS Code, etc.) *before* signing, otherwise the typed letters go into wherever your cursor was when you started.

### Controls

- `Q` — quit
- `SPACE` — insert a space manually
- `BACKSPACE` — delete last character

### Configuration (top of `webcam.py`)

| Variable | Default | Notes |
|---|---|---|
| `CAMERA_INDEX` | `0` | Change to `1` if you have an external webcam and want to use that one |
| `CONFIDENCE` | `0.85` | Minimum prediction confidence before a letter is accepted |
| `COOLDOWN` | `3` | Seconds between consecutive typed letters |
| `BOX_SIZE` | `350` | Pixel size of the hand capture box |

### Expected accuracy

The webcam runs the model on real-world conditions (your lighting, your hand, your background) which differ substantially from the training data. Expect accuracy closer to the **77.89% external benchmark** than the 99.85% internal one. This is the central finding documented in our report.

---

## Datasets

| Use | Source | Size |
|---|---|---|
| Training / validation / internal test | [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | 87,000 images, 29 classes |
| External generalisation test | [debashishsau/aslamerican-sign-language-aplhabet-dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) | capped at 500 / class = 13,500 images |

Both datasets are downloaded automatically by the code via `kagglehub` — no manual download needed. The training dataset uses 27 of its 29 classes; J and Z are **excluded** because they require motion to disambiguate, which is outside the scope of static-image recognition.

---

## Dependencies

See `requirements.txt`. Core stack:

```
numpy
torch>=2.0
scikit-learn
opencv-python
kagglehub
pyautogui     # only needed for webcam.py
```

Tested on Python 3.10+ and PyTorch 2.x. Developed and trained on Google Colab with a T4 GPU.

---

## Reproducibility notes

- **Random seeds.** `DataPreprocessing.py` uses `random_state=2026` for the train/val/test split — splits are deterministic across re-runs.
- **Model checkpoints.** `train.py` saves *every epoch* to `asl_cnn.pth` and *the best validation epoch* separately to `asl_cnn_best.pth`. Evaluation always uses `asl_cnn_best.pth`.
- **Crash safety.** Each stage persists its outputs to disk before moving on, so a Colab disconnect never costs more than one epoch of work.
- **Stable to ±0.5pp across re-runs** — we don't seed CUDA/cuDNN internals, so metrics fluctuate slightly. The headline numbers (99.85% internal, 77.89% external) are reproducible to that precision.

---

## Team

Santiago Landinez, Federico Mendez, Isabella Garay, María Fernanda Jacobo, Cesar Prieto, Faris Alami
*(add your name in alphabetical position)*

IE University, BDBA 2025 — Machine Learning Foundations.

---

## Acknowledgements

- Kaggle dataset authors `grassknoted` and `debashishsau`.
- Built with [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/), [OpenCV](https://opencv.org/), and [kagglehub](https://github.com/Kaggle/kagglehub).
