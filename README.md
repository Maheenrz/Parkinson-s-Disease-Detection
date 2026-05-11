# Parkinson's Disease Detection via Gait Analysis

A machine learning project that detects Parkinson's Disease from sensor-based walking data comparing classical ML, deep neural networks, and LSTM sequence models to find the most effective diagnostic approach.

---

## What This Project Does

Parkinson's Disease subtly changes the way people walk their step rhythm becomes irregular and asymmetrical. This project uses pressure sensor data from the feet (Vertical Ground Reaction Force / VGRF) to train classifiers that can distinguish between healthy individuals and Parkinson's patients.

The full ML pipeline:
1. Load and parse raw PhysioNet gait files
2. Preprocess and clean the time-series data
3. Engineer biomechanical features (asymmetry, step count, variability)
4. Compare 5 classical ML models
5. Train a Fully Connected DNN on raw windows
6. Train an LSTM on sequential gait data
7. Compare all three approaches side by side

---

## Dataset

**Name:** [Gait in Parkinson's Disease Database](https://physionet.org/content/gaitpdb/1.0.0/)  
**Source:** PhysioNet (open access)  
**Citation:** Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.

| Detail | Value |
|---|---|
| Sensor type | Force plates / pressure insoles |
| Measurement | Vertical Ground Reaction Force (VGRF) |
| Sensors | 8 per foot (16 total) |
| Sampling rate | 100 Hz |
| Subjects | 93 Parkinson's patients + 73 healthy controls |
| Columns per file | 19 (time, L1-L8, R1-R8, total_left, total_right) |

---

## Methodology

### Phase 1 — Data Loading & Preprocessing
- Custom `PhysioNetGaitLoader` parses filenames to extract subject ID, diagnosis (Control/Parkinson), and trial number
- Preprocessing pipeline: timestamp parsing → chronological sorting → missing value interpolation → duplicate removal → negative force correction → outlier clipping (±4σ)

### Phase 2 — Feature Engineering + Classical ML
- Signal segmented into **3-second windows with 50% overlap**
- Features extracted per window:
  - Time-domain stats: Mean, Std, Max, Skewness
  - Clinical biomarkers: Force Asymmetry Index, Variability Asymmetry, Step Count, Step Symmetry
- **Subject-wise 70/30 train-test split** (prevents data leakage — different subjects in train vs. test)
- Five models compared: Logistic Regression, SVM (RBF), Decision Tree, Random Forest, XGBoost

### Phase 3 — Fully Connected DNN
- Raw windows flattened (100 timesteps x 16 sensors = 1600 features) fed into a Dense network
- Architecture: 256 → Dropout → 128 → Dropout → 64 → Sigmoid

### Phase 4 — LSTM Sequence Model
- 3D sequences (samples x 100 timesteps x 18 features) fed directly into LSTM
- Architecture: LSTM(64) → Dropout → Dense(32) → Dropout → Sigmoid

---

## Results

| Approach | Accuracy | F1-Score |
|---|---|---|
| **Random Forest** (engineered features) | **0.7708** | **0.7281** |
| Fully Connected DNN (raw windows) | 0.6729 | 0.6362 |
| LSTM (sequential) | - | - |

Random Forest came out on top. Feature engineering — particularly the asymmetry metrics — carries more diagnostic signal than raw deep learning on a dataset of this size.

---

## Project Structure

```
parkinsons-gait-detection/
│
├── Parkinson's_Disease_Detection.ipynb   # Main notebook (all phases)
│
├── data/
│   └── raw/                                  # Place .txt gait files here
│       ├── GaCo01_01.txt
│       ├── GaPt01_01.txt
│       └── ...
│
└── README.md
```

---

## Setup & Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/parkinsons-gait-detection.git
cd parkinsons-gait-detection
```

### Step 2 — Download the Dataset

1. Go to [https://physionet.org/content/gaitpdb/1.0.0/](https://physionet.org/content/gaitpdb/1.0.0/)
2. Create a free PhysioNet account and log in
3. Download all `.txt` files
4. Place them inside `data/raw/` in the project folder

> Do **not** include `format.txt` — the loader automatically skips it, but it doesn't belong in the data folder.

### Step 3 — Choose Your Environment

#### Option A: Google Colab (Recommended)

This notebook was built and tested on Colab. It's the easiest way to run it.

1. Upload the notebook to Google Colab or open it via GitHub:
   `File → Open notebook → GitHub → paste your repo URL`

2. Upload the `data/raw/` folder to your Google Drive under `MyDrive/data/`

3. The notebook mounts Drive automatically:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Run all cells top to bottom (`Runtime → Run all`)

#### Option B: Local Environment

1. Python 3.8+ required

2. Install dependencies:

```bash
pip install pandas numpy scipy scikit-learn xgboost tensorflow matplotlib seaborn
```

3. Open the notebook:

```bash
jupyter notebook BITF22M034_BITF22M023_Project_ML.ipynb
```

4. In the data loading section, `data_path` is already set to `'data/raw'` — it works as-is if you followed the structure above.

5. Run all cells in order.

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading and manipulation |
| `scipy` | Signal processing |
| `scikit-learn` | Classical ML models, preprocessing, metrics |
| `xgboost` | XGBoost classifier |
| `tensorflow / keras` | DNN and LSTM models |
| `matplotlib`, `seaborn` | Visualizations and confusion matrices |

---

## Key Design Decisions

- **Subject-wise split** instead of random row split — the scientifically correct approach for medical ML. Mixing windows from the same patient into both train and test sets inflates scores artificially.
- **Macro F1 as primary metric** — the dataset is moderately imbalanced, so accuracy alone is misleading.
- **Feature engineering outperforms raw DNN** — the dataset has ~166 subjects, which is small for deep learning. Handcrafted asymmetry features encode domain knowledge the DNN would need far more data to discover on its own.

---

## Citation

If you use this work or the dataset, please cite:

```
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE,
Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet:
Components of a New Research Resource for Complex Physiologic Signals.
Circulation 101(23):e215-e220 (2000).
```
