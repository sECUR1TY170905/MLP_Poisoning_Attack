# 🧠 MLP Poisoning Attack — Variant 9

> **Project ML | PTIT | Attack on Artificial Intelligence (Data Poisoning)**  
> Variant parameters: `N = 50%` · `b = 2` · `dataset = number-3.txt`

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Algorithms](#-algorithms)
- [Experiments & Steps](#-experiments--steps)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Files](#-output-files)

---

## 🔍 Overview

This project studies **data poisoning attacks** against machine learning classifiers — a form of adversarial attack where an attacker deliberately corrupts the training data to degrade model performance.

Two types of poisoning attacks are investigated:

| Attack Type | Description |
|---|---|
| **Label-Flip** | Randomly inverts the class labels (0↔1) of a percentage of training samples |
| **Feature-Shuffle** | Randomly shuffles the feature values within selected training samples |

Two classifiers are implemented and compared side-by-side:

- **MLP** (Multi-Layer Perceptron) — `variant9_ml_project.py`
- **XGBoost** (Gradient Boosted Trees) — `variant9_xgboost.py`

---

## 📁 Project Structure

```
MLP_Poisoning_Attack/
│
├── dataset/
│   └── number-3.txt              # Health indicator dataset (Variant 9)
│
├── variant9_ml_project.py        # Main experiment — MLP Classifier
├── variant9_xgboost.py           # Step 15 extension — XGBoost Classifier
│
├── picture1_clean_vs_poisoned.png
├── picture2_decreasing_poison.png
├── picture3_labelflip_vs_featureshuffle.png
├── picture4_training_loss.png
│
├── xgb_picture1_clean_vs_poisoned.png
├── xgb_picture2_decreasing_poison.png
├── xgb_picture3_labelflip_vs_featureshuffle.png
├── xgb_picture4_training_loss.png
│
├── number-1.txt                  # (auxiliary data)
├── number-2.txt                  # (auxiliary data)
├── Project_ML_task.docx          # Assignment specification
└── README.md
```

---

## 📊 Dataset

- **File:** `dataset/number-3.txt`
- **Format:** Each row = one sample — 16 numerical features + 1 binary label
- **Features:** 16 health indicators per patient
- **Labels:** `1.0` = Healthy · `0.0` = Sick
- **Split:** 80% Train / 20% Test (stratified, `random_state=42`)
- **Preprocessing:** StandardScaler (zero mean, unit variance)

---

## 🤖 Algorithms

### MLP Classifier (`variant9_ml_project.py`)
```
Architecture:  Input(16) → Dense(64) → Dense(32) → Dense(16) → Output(1)
Activation:    ReLU
Optimizer:     Adam  (lr = 1e-3)
Regularization: L2 alpha = 1e-4
Epochs:        100
Batch size:    64
```

### XGBoost Classifier (`variant9_xgboost.py`)
```
n_estimators:   100
learning_rate:  0.1
max_depth:      4
subsample:      0.8
colsample_bytree: 0.8
eval_metric:    logloss
```

---

## 🧪 Experiments & Steps

| Step | Description | Poison % |
|------|-------------|----------|
| **1** | Load dataset `number-3.txt` | — |
| **2** | Train/Test split 80/20 (stratified) | — |
| **3** | Train clean MLP baseline | 0% |
| **4–6** | Label-flip poisoning | **N = 50%** |
| **7–9** | Decreasing poison `b=2`: 50% → 40% → 30% → 20% | 50/40/30/20% |
| **10** | Tiny poisoning: 1% and 1 single row | 1% / 1 row |
| **11** ⭐ | Extreme poisoning: 99% and 100% | 99% / 100% |
| **12–14** | Feature-shuffle vs Label-flip comparison | 50/40/30/20/1% |
| **15** | Repeat all steps with **XGBoost** | all of the above |

---

## 📈 Results

### MLP — Output Charts

| Chart | Description |
|-------|-------------|
| `picture1_clean_vs_poisoned.png` | Clean model vs. 50% poisoned accuracy |
| `picture2_decreasing_poison.png` | Accuracy across decreasing poison rates (b=2) |
| `picture3_labelflip_vs_featureshuffle.png` | Label-flip vs. Feature-shuffle side-by-side |
| `picture4_training_loss.png` | Training loss curves: clean vs. poisoned |

### XGBoost — Output Charts

| Chart | Description |
|-------|-------------|
| `xgb_picture1_clean_vs_poisoned.png` | [XGBoost] Clean vs. 50% poisoned accuracy |
| `xgb_picture2_decreasing_poison.png` | [XGBoost] Accuracy vs. decreasing poison |
| `xgb_picture3_labelflip_vs_featureshuffle.png` | [XGBoost] Label-flip vs. Feature-shuffle |
| `xgb_picture4_training_loss.png` | [XGBoost] Training loss: clean vs. poisoned |

### Key Observations

- **Label-flip** is far more damaging than **feature-shuffle** — scrambling labels confuses the model much more than scrambling features.
- At **50% label-flip**, accuracy drops dramatically from the clean baseline.
- At **99–100% poison**, the model's predictions become nearly **inverted** (accuracy approaches 0%), effectively becoming adversarial.
- As the poisoning rate **decreases** (50% → 20%), accuracy gradually recovers toward the clean baseline.
- **XGBoost** generally shows stronger **robustness** against label-flip poisoning compared to MLP due to its ensemble nature.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/sECUR1TY170905/MLP_Poisoning_Attack.git
cd MLP_Poisoning_Attack

# Install dependencies
pip install numpy scikit-learn xgboost matplotlib rich
```

**Requirements:**

| Package | Version |
|---------|---------|
| `numpy` | ≥ 1.24 |
| `scikit-learn` | ≥ 1.3 |
| `xgboost` | ≥ 2.0 |
| `matplotlib` | ≥ 3.7 |
| `rich` | ≥ 13.0 |

---

## 🚀 Usage

### Run MLP experiments (Steps 1–14):
```bash
python variant9_ml_project.py
```

### Run XGBoost experiments (Step 15):
```bash
python variant9_xgboost.py
```

Both scripts will:
1. Print a rich, color-coded progress output to the console
2. Save all result charts as `.png` files in the project root

---

## 📤 Output Files

After running both scripts, the following PNG files will be generated:

```
picture1_clean_vs_poisoned.png
picture2_decreasing_poison.png
picture3_labelflip_vs_featureshuffle.png
picture4_training_loss.png

xgb_picture1_clean_vs_poisoned.png
xgb_picture2_decreasing_poison.png
xgb_picture3_labelflip_vs_featureshuffle.png
xgb_picture4_training_loss.png
```

---

## 👤 Author

**Variant 9** — PTIT Machine Learning Project  
📅 June–July 2026

---

> *"The best defense against data poisoning is understanding how it works."*
