
# =============================================================================
# PROJECT ML - VARIANT 9
# Attack on Artificial Intelligence (Data Poisoning)
# N = 50%, b = 2, file = number-3.txt
# =============================================================================
# Variant 9 parameters:
#   N  = 50%  -> initial poisoning percentage
#   b  = 2    -> DECREASE poisoning by 10/20/30% in step 7-9
#   file = number-3.txt
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')   # non-interactive backend – no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import warnings
warnings.filterwarnings('ignore')

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.edgecolor':   '#0f3460',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#e0e0e0',
    'ytick.color':      '#e0e0e0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#0f3460',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})
COLORS = ['#00d4ff', '#ff6b6b', '#ffd166', '#06d6a0', '#e040fb', '#ff9800']

# =============================================================================
# STEP 1 – Load dataset
# =============================================================================
print("=" * 60)
print("STEP 1: Loading dataset (number-3.txt)")
print("=" * 60)

DATA_FILE = 'number-3.txt'
data = []
with open(DATA_FILE, 'r') as f:
    for line in f:
        vals = list(map(float, line.strip().split()))
        data.append(vals)

data = np.array(data)
X = data[:, :16]   # 16 health indicators
y = data[:, 16]    # label: 1.0 = healthy, 0.0 = sick

print(f"  Total samples      : {len(data)}")
print(f"  Features per sample: {X.shape[1]}")
print(f"  Class distribution : Healthy(1.0)={int(y.sum())}  |  Sick(0.0)={int((y==0).sum())}")

# =============================================================================
# STEP 2 – Split 80/20 with stratification
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Train/Test split (80/20, stratified)")
print("=" * 60)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples    : {len(X_test)}")
print(f"  Train class dist: Healthy={int(y_train.sum())}  Sick={int((y_train==0).sum())}")
print(f"  Test  class dist: Healthy={int(y_test.sum())}   Sick={int((y_test==0).sum())}")

# ── Tensor helpers ────────────────────────────────────────────────────────────
def to_tensors(X, y):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return Xt, yt

# =============================================================================
# STEP 3 – Define MLP: input(16) -> 64 -> 32 -> 16 -> output(1)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Define & train MLP (16->64->32->16->1)")
print("=" * 60)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16,  1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_model(X_tr, y_tr, epochs=100, lr=1e-3, batch_size=64, verbose=True):
    """Train MLP and return (model, train_loss_history)."""
    Xt, yt = to_tensors(X_tr, y_tr)
    loader  = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model     = MLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        losses.append(epoch_loss / len(Xt))
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={losses[-1]:.4f}")

    return model, losses


def evaluate(model, X_ev, y_ev):
    """Return accuracy (%) on evaluation set."""
    model.eval()
    with torch.no_grad():
        Xt, yt = to_tensors(X_ev, y_ev)
        preds  = (model(Xt) >= 0.5).float()
        acc    = (preds == yt).float().mean().item() * 100
    return acc


# Train on CLEAN data
print("  Training on CLEAN dataset…")
model_clean, clean_losses = train_model(X_train, y_train)
acc_clean = evaluate(model_clean, X_test, y_test)
print(f"\n  [OK] Accuracy (clean model): {acc_clean:.2f}%")

# =============================================================================
# STEPS 4-6 – Poison training set (label flip, N=50%)
# =============================================================================
print("\n" + "=" * 60)
print("STEPS 4-6: Label-flip poisoning @ N=50%")
print("=" * 60)


def poison_labels(X_tr, y_tr, pct):
    """Invert class labels for `pct`% of training rows."""
    X_p = X_tr.copy()
    y_p = y_tr.copy()
    n_poison = max(1, int(len(y_p) * pct / 100))
    idx = np.random.choice(len(y_p), n_poison, replace=False)
    y_p[idx] = 1.0 - y_p[idx]   # flip 0↔1
    return X_p, y_p, n_poison


np.random.seed(42)
X_p50, y_p50, n50 = poison_labels(X_train, y_train, 50)
print(f"  Poisoned {n50} samples (50% of {len(y_train)})")
print("  Training on POISONED dataset…")
model_p50, p50_losses = train_model(X_p50, y_p50)
acc_p50 = evaluate(model_p50, X_test, y_test)
print(f"\n  [OK] Accuracy (50% poisoned): {acc_p50:.2f}%")
print(f"  [DROP] Drop: {acc_clean - acc_p50:.2f}%")

# ── Picture 1 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['Clean model', 'Poisoned 50%'], [acc_clean, acc_p50],
              color=[COLORS[0], COLORS[1]], width=0.4, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, [acc_clean, acc_p50]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold', color='white')
ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Picture 1 – Clean vs. Poisoned (50%) Accuracy', fontsize=14, fontweight='bold', color='#00d4ff')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig('picture1_clean_vs_poisoned.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: picture1_clean_vs_poisoned.png")

# =============================================================================
# STEPS 7-9 – b=2 -> DECREASE poisoning by 10/20/30 percent
# =============================================================================
print("\n" + "=" * 60)
print("STEPS 7-9: b=2 -> Decrease poisoning (40%, 30%, 20%)")
print("=" * 60)

pcts_b2  = [50, 40, 30, 20]   # 50-10, 50-20, 50-30
accs_b2  = [acc_p50]
labels_b2 = ['50%']

for pct in [40, 30, 20]:
    np.random.seed(42)
    Xp, yp, n = poison_labels(X_train, y_train, pct)
    print(f"\n  Poisoning @ {pct}%  ({n} samples) …")
    m, _ = train_model(Xp, yp, verbose=False)
    a = evaluate(m, X_test, y_test)
    accs_b2.append(a)
    labels_b2.append(f'{pct}%')
    print(f"  [OK] Accuracy: {a:.2f}%")

print(f"\n  Clean baseline: {acc_clean:.2f}%")
accs_b2_full  = [acc_clean] + accs_b2
labels_b2_full = ['Clean(0%)'] + labels_b2

# ── Picture 2 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels_b2_full))
bars = ax.bar(x, accs_b2_full, color=COLORS[:len(x)], width=0.5, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, accs_b2_full):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')
ax.set_xticks(x)
ax.set_xticklabels(labels_b2_full, fontsize=11)
ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Picture 2 – Accuracy vs. Poisoning % (b=2: decreasing)', fontsize=14, fontweight='bold', color='#00d4ff')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig('picture2_decreasing_poison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: picture2_decreasing_poison.png")

# =============================================================================
# STEP 10 – Poison at 1% and 1 single row
# =============================================================================
print("\n" + "=" * 60)
print("STEP 10: Tiny poisoning (1% and 1 single row)")
print("=" * 60)

np.random.seed(42)
Xp1pct, yp1pct, n1pct = poison_labels(X_train, y_train, 1)
print(f"  Poisoning @ 1%  ({n1pct} samples) …")
m1pct, _ = train_model(Xp1pct, yp1pct, verbose=False)
acc_1pct  = evaluate(m1pct, X_test, y_test)
print(f"  [OK] Accuracy: {acc_1pct:.2f}%")

# Single row
np.random.seed(42)
Xp1row, yp1row, _ = poison_labels(X_train, y_train, 0.0)   # no poison
y_1row = yp1row.copy()
y_1row[0] = 1.0 - y_1row[0]                                 # flip exactly 1 row
print(f"\n  Poisoning 1 single row …")
m1row, _ = train_model(Xp1row, y_1row, verbose=False)
acc_1row  = evaluate(m1row, X_test, y_test)
print(f"  [OK] Accuracy: {acc_1row:.2f}%")

# =============================================================================
# STEP 11 (*) – Extreme poisoning 99% and 100%
# =============================================================================
print("\n" + "=" * 60)
print("STEP 11 (*): Extreme poisoning (99% and 100%)")
print("=" * 60)

for pct in [99, 100]:
    np.random.seed(42)
    Xp, yp, n = poison_labels(X_train, y_train, pct)
    print(f"\n  Poisoning @ {pct}%  ({n} samples) …")
    m, _ = train_model(Xp, yp, verbose=False)
    a = evaluate(m, X_test, y_test)
    print(f"  [OK] Accuracy: {a:.2f}%  <- {'inverted!' if a < 40 else 'surprising?'}")

# =============================================================================
# STEPS 12-14 – Feature-shuffle poisoning
# =============================================================================
print("\n" + "=" * 60)
print("STEPS 12-14: Feature-shuffle poisoning")
print("=" * 60)


def poison_features(X_tr, y_tr, pct):
    """Shuffle the order of features in `pct`% of training rows."""
    X_p = X_tr.copy()
    y_p = y_tr.copy()
    n_poison = max(1, int(len(y_p) * pct / 100))
    idx = np.random.choice(len(y_p), n_poison, replace=False)
    for i in idx:
        np.random.shuffle(X_p[i])   # in-place feature shuffle
    return X_p, y_p, n_poison


pcts_feat   = [50, 40, 30, 20, 1]
accs_feat   = []
accs_label  = []

print("\n  Comparing label-flip vs feature-shuffle:")
print(f"  {'Poison%':>8}  {'Label-flip Acc':>16}  {'Feature-shuffle Acc':>20}")
print("  " + "-" * 50)

for pct in pcts_feat:
    np.random.seed(42)
    Xpl, ypl, _  = poison_labels(X_train, y_train, pct)
    ml, _         = train_model(Xpl, ypl, verbose=False)
    al            = evaluate(ml, X_test, y_test)

    np.random.seed(42)
    Xpf, ypf, _  = poison_features(X_train, y_train, pct)
    mf, _         = train_model(Xpf, ypf, verbose=False)
    af            = evaluate(mf, X_test, y_test)

    accs_label.append(al)
    accs_feat.append(af)
    print(f"  {pct:>7}%  {al:>15.2f}%  {af:>19.2f}%")

# ── Comparison chart ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x  = np.arange(len(pcts_feat))
w  = 0.35
b1 = ax.bar(x - w/2, accs_label, w, label='Label-flip',       color=COLORS[1], edgecolor='white')
b2 = ax.bar(x + w/2, accs_feat,  w, label='Feature-shuffle', color=COLORS[2], edgecolor='white')
ax.axhline(acc_clean, color=COLORS[0], linestyle='--', linewidth=1.5, label=f'Clean ({acc_clean:.1f}%)')
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, color='white')
ax.set_xticks(x)
ax.set_xticklabels([f'{p}%' for p in pcts_feat])
ax.set_ylim(0, 115)
ax.set_xlabel('Poisoning percentage', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Steps 12-14 – Label-flip vs. Feature-shuffle Poisoning', fontsize=14, fontweight='bold', color='#00d4ff')
ax.legend(fontsize=11)
ax.grid(axis='y')
plt.tight_layout()
plt.savefig('picture3_labelflip_vs_featureshuffle.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: picture3_labelflip_vs_featureshuffle.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY – VARIANT 9")
print("=" * 60)
print(f"  {'Experiment':<35} {'Accuracy':>10}")
print("  " + "-" * 47)
print(f"  {'Clean model':<35} {acc_clean:>9.2f}%")
print(f"  {'Label-flip 50% (N=50%)':<35} {acc_p50:>9.2f}%")
for lbl, acc in zip(labels_b2, accs_b2):
    print(f"  {'Label-flip ' + lbl:<35} {acc:>9.2f}%")
print(f"  {'Label-flip 1%':<35} {acc_1pct:>9.2f}%")
print(f"  {'Label-flip 1 row':<35} {acc_1row:>9.2f}%")
print("=" * 60)

# ── Training loss curves (clean vs 50% poisoned) ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(clean_losses, color=COLORS[0], linewidth=2, label='Clean training loss')
ax.plot(p50_losses,   color=COLORS[1], linewidth=2, label='Poisoned 50% training loss')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('BCE Loss', fontsize=12)
ax.set_title('Training Loss: Clean vs. Poisoned (50%)', fontsize=14, fontweight='bold', color='#00d4ff')
ax.legend(fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.savefig('picture4_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: picture4_training_loss.png")

print("\n[DONE] ALL STEPS COMPLETE! Check the saved PNG files for plots.")
