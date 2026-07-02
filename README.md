# MLP Poisoning Attack – Variant 9

> **Project ML – Attack on Artificial Intelligence (Data Poisoning)**  
> Course: Machine Learning | PTIT Vietnam

## Overview

This project simulates and analyzes **data poisoning attacks** on an MLP (Multi-Layer Perceptron) binary classifier trained on health data.

- **Dataset**: `number-3.txt` — 2000 samples, 16 health-indicator features, binary label (Healthy=1 / Sick=0)
- **Model**: MLP `16 → 64 → 32 → 16 → 1` with BatchNorm, Dropout, Adam optimizer
- **Variant parameters**: N=50%, b=2 (decreasing)

## Experiment Steps

| Step | Description |
|------|-------------|
| 1–2 | Load dataset & 80/20 stratified split |
| 3 | Train & evaluate **clean** model |
| 4–6 | **Label-flip poisoning @ 50%** — flip class labels of 50% training samples |
| 7–9 | **b=2 (decreasing)**: reduce poisoning from 50% → 40% → 30% → 20% |
| 10 | **Tiny poisoning**: 1% and 1 single row |
| 11 | **Extreme poisoning**: 99% and 100% |
| 12–14 | **Feature-shuffle poisoning** vs label-flip comparison |

## Results Summary

| Experiment | Accuracy |
|------------|----------|
| Clean model | ~88% |
| Label-flip 50% | ~46% (drop ~42%) |
| Label-flip 40% | ~62% |
| Label-flip 30% | ~76% |
| Label-flip 20% | ~83% |
| Label-flip 1% | ~87% |
| Label-flip 1 row | ~87% |
| Extreme 99% | ~13% ← inverted! |
| Extreme 100% | ~13% ← inverted! |

## Key Finding

**Label-flip poisoning** is far more destructive than **feature-shuffle poisoning**.  
At 50% poisoning: label-flip → ~46% accuracy vs feature-shuffle → ~84% accuracy.

## Output Plots

| File | Content |
|------|---------|
| `picture1_clean_vs_poisoned.png` | Clean vs Poisoned 50% accuracy |
| `picture2_decreasing_poison.png` | Accuracy vs decreasing poison % (b=2) |
| `picture3_labelflip_vs_featureshuffle.png` | Label-flip vs Feature-shuffle comparison |
| `picture4_training_loss.png` | Training loss curves: clean vs poisoned |

## How to Run

```bash
pip install torch numpy scikit-learn matplotlib
python variant9_ml_project.py
```

> Plots are saved as PNG files (no GUI window required).
