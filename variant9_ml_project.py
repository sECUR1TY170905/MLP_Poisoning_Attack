
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
matplotlib.use('Agg')   # non-interactive backend - no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import warnings
warnings.filterwarnings('ignore')

# ── Rich console ──────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich import box
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule

console = Console()

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
# STEP 1 - Load dataset
# =============================================================================
console.print()
console.print(Panel("[bold cyan]STEP 1 : Loading Dataset  [white]number-3.txt[/white][/bold cyan]",
                    style="bold blue", box=box.DOUBLE_EDGE))

DATA_FILE = 'number-3.txt'
data = []
with open(DATA_FILE, 'r') as f:
    for line in f:
        vals = list(map(float, line.strip().split()))
        data.append(vals)

data = np.array(data)
X = data[:, :16]   # 16 health indicators
y = data[:, 16]    # label: 1.0 = healthy, 0.0 = sick

t1 = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
t1.add_column("Key",   style="bold bright_white", no_wrap=True)
t1.add_column("Value", style="bright_green")
t1.add_row("Total samples",       str(len(data)))
t1.add_row("Features per sample", str(X.shape[1]))
t1.add_row("Healthy (1.0)",       str(int(y.sum())))
t1.add_row("Sick    (0.0)",       str(int((y == 0).sum())))
console.print(t1)

# =============================================================================
# STEP 2 - Split 80/20 with stratification
# =============================================================================
console.print(Panel("[bold cyan]STEP 2 : Train / Test Split  [white]80% / 20%  |  Stratified[/white][/bold cyan]",
                    style="bold blue", box=box.DOUBLE_EDGE))

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

t2 = Table(show_header=True, box=box.SIMPLE_HEAVY, padding=(0, 2))
t2.add_column("Split", style="bold bright_white", justify="left")
t2.add_column("Samples",  style="bright_cyan",  justify="right")
t2.add_column("Healthy",  style="bright_green", justify="right")
t2.add_column("Sick",     style="bright_red",   justify="right")
t2.add_row("Train", str(len(X_train)), str(int(y_train.sum())), str(int((y_train == 0).sum())))
t2.add_row("Test",  str(len(X_test)),  str(int(y_test.sum())),  str(int((y_test  == 0).sum())))
console.print(t2)

# ── Tensor helpers ────────────────────────────────────────────────────────────
def to_tensors(X, y):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return Xt, yt

# =============================================================================
# STEP 3 - Define MLP: input(16) -> 64 -> 32 -> 16 -> output(1)
# =============================================================================
console.print(Panel("[bold cyan]STEP 3 : Define & Train MLP  [white]16 -> 64 -> 32 -> 16 -> 1[/white][/bold cyan]",
                    style="bold blue", box=box.DOUBLE_EDGE))

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


def train_model(X_tr, y_tr, epochs=100, lr=1e-3, batch_size=64,
                verbose=True, label="Training"):
    """Train MLP and return (model, train_loss_history)."""
    Xt, yt  = to_tensors(X_tr, y_tr)
    loader  = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model     = MLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    losses = []

    if verbose:
        with Progress(
            TextColumn("  [progress.description]{task.description}"),
            BarColumn(bar_width=30, style="cyan", complete_style="bright_cyan"),
            TextColumn("[bold white]{task.percentage:>5.1f}%"),
            TextColumn("[dim]loss=[bold yellow]{task.fields[loss]:.4f}[/bold yellow]"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(label, total=epochs, loss=0.0)
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
                progress.update(task, advance=1, loss=losses[-1])
    else:
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
console.print("  [bold white]Training on [bright_green]CLEAN[/bright_green] dataset...[/bold white]")
model_clean, clean_losses = train_model(X_train, y_train, label="Clean   ")
acc_clean = evaluate(model_clean, X_test, y_test)
console.print(f"  [bold bright_green]ACCURACY (clean model) : {acc_clean:.2f}%[/bold bright_green]\n")

def fmt_diff(a):
    """Format accuracy difference vs clean baseline with color."""
    d = a - acc_clean
    if d > 0:
        return f"[bold bright_green]Diff vs Clean : +{d:.2f}%[/bold bright_green]"
    elif d < 0:
        return f"[bold bright_red]Diff vs Clean :  {d:.2f}%[/bold bright_red]"
    else:
        return f"[bold white]Diff vs Clean :  0.00% (no change)[/bold white]"

# =============================================================================
# STEPS 4-6 - Poison training set (label flip, N=50%)
# =============================================================================
console.print(Panel("[bold cyan]STEPS 4-6 : Label-Flip Poisoning  [white]N = 50%[/white][/bold cyan]",
                    style="bold blue", box=box.DOUBLE_EDGE))


def poison_labels(X_tr, y_tr, pct):
    """Invert class labels for `pct`% of training rows."""
    X_p = X_tr.copy()
    y_p = y_tr.copy()
    n_poison = max(1, int(len(y_p) * pct / 100))
    idx = np.random.choice(len(y_p), n_poison, replace=False)
    y_p[idx] = 1.0 - y_p[idx]   # flip 0<->1
    return X_p, y_p, n_poison


np.random.seed(42)
X_p50, y_p50, n50 = poison_labels(X_train, y_train, 50)
console.print(f"  [yellow]Poisoned [bold]{n50}[/bold] samples  (50% of {len(y_train)})[/yellow]")
console.print("  [bold white]Training on [bright_red]POISONED[/bright_red] dataset...[/bold white]")
model_p50, p50_losses = train_model(X_p50, y_p50, label="Poison 50%")
acc_p50 = evaluate(model_p50, X_test, y_test)
console.print(f"  [bold bright_red]ACCURACY (50% poisoned) : {acc_p50:.2f}%[/bold bright_red]")
console.print(f"  {fmt_diff(acc_p50)}\n")

# ── Picture 1 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['Clean model', 'Poisoned 50%'], [acc_clean, acc_p50],
              color=[COLORS[0], COLORS[1]], width=0.4, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, [acc_clean, acc_p50]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold', color='white')
ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Picture 1 - Clean vs. Poisoned (50%) Accuracy', fontsize=14, fontweight='bold', color='#00d4ff')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig('picture1_clean_vs_poisoned.png', dpi=150, bbox_inches='tight')
plt.close()
console.print("  [dim]Saved: picture1_clean_vs_poisoned.png[/dim]")

# =============================================================================
# STEPS 7-9 - b=2 -> DECREASE poisoning by 10/20/30 percent
# =============================================================================
console.print()
console.print(Panel("[bold cyan]STEPS 7-9 : Decreasing Poison  [white]b=2  ->  50% -> 40% -> 30% -> 20%[/white][/bold cyan]",
                    style="bold blue", box=box.DOUBLE_EDGE))

pcts_b2   = [50, 40, 30, 20]
accs_b2   = [acc_p50]
labels_b2 = ['50%']

for pct in [40, 30, 20]:
    np.random.seed(42)
    Xp, yp, n = poison_labels(X_train, y_train, pct)
    console.print(f"  [yellow]Poisoning @ [bold]{pct}%[/bold]  ({n} samples)...[/yellow]")
    m, _ = train_model(Xp, yp, verbose=False)
    a = evaluate(m, X_test, y_test)
    accs_b2.append(a)
    labels_b2.append(f'{pct}%')
    console.print(f"  [bright_green]Accuracy : {a:.2f}%[/bright_green]  |  {fmt_diff(a)}")

console.print(f"\n  [bright_white]Clean baseline : [bold green]{acc_clean:.2f}%[/bold green][/bright_white]")
accs_b2_full   = [acc_clean] + accs_b2
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
ax.set_title('Picture 2 - Accuracy vs. Poisoning % (b=2: decreasing)', fontsize=14, fontweight='bold', color='#00d4ff')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig('picture2_decreasing_poison.png', dpi=150, bbox_inches='tight')
plt.close()
console.print("  [dim]Saved: picture2_decreasing_poison.png[/dim]")

# =============================================================================
# STEP 10 - Poison at 1% and 1 single row
# =============================================================================
console.print()
console.print(Panel("[bold cyan]STEP 10 : Tiny Poisoning  [white]1%  and  1 single row[/white][/bold cyan]",
                    style="bold blue", box=box.DOUBLE_EDGE))

np.random.seed(42)
Xp1pct, yp1pct, n1pct = poison_labels(X_train, y_train, 1)
console.print(f"  [yellow]Poisoning @ 1%  ({n1pct} samples)...[/yellow]")
m1pct, _ = train_model(Xp1pct, yp1pct, verbose=False)
acc_1pct  = evaluate(m1pct, X_test, y_test)
console.print(f"  [bright_green]Accuracy : {acc_1pct:.2f}%[/bright_green]  |  {fmt_diff(acc_1pct)}")

# Single row
np.random.seed(42)
Xp1row, yp1row, _ = poison_labels(X_train, y_train, 0.0)   # no poison
y_1row = yp1row.copy()
y_1row[0] = 1.0 - y_1row[0]                                 # flip exactly 1 row
console.print(f"\n  [yellow]Poisoning 1 single row...[/yellow]")
m1row, _ = train_model(Xp1row, y_1row, verbose=False)
acc_1row  = evaluate(m1row, X_test, y_test)
console.print(f"  [bright_green]Accuracy : {acc_1row:.2f}%[/bright_green]  |  {fmt_diff(acc_1row)}")

# =============================================================================
# STEP 11 (*) - Extreme poisoning 99% and 100%
# =============================================================================
console.print()
console.print(Panel("[bold cyan]STEP 11 (*) : Extreme Poisoning  [white]99%  and  100%[/white][/bold cyan]",
                    style="bold blue", box=box.DOUBLE_EDGE))

for pct in [99, 100]:
    np.random.seed(42)
    Xp, yp, n = poison_labels(X_train, y_train, pct)
    console.print(f"  [yellow]Poisoning @ [bold]{pct}%[/bold]  ({n} samples)...[/yellow]")
    m, _ = train_model(Xp, yp, verbose=False)
    a = evaluate(m, X_test, y_test)
    tag = "[bold red]<-- INVERTED![/bold red]" if a < 40 else "[bold yellow]<-- surprising?[/bold yellow]"
    console.print(f"  [bright_red]Accuracy : {a:.2f}%[/bright_red]  {tag}")
    console.print(f"  {fmt_diff(a)}")

# =============================================================================
# STEPS 12-14 - Feature-shuffle poisoning
# =============================================================================
console.print()
console.print(Panel("[bold cyan]STEPS 12-14 : Feature-Shuffle Poisoning  [white]vs  Label-Flip[/white][/bold cyan]",
                    style="bold blue", box=box.DOUBLE_EDGE))


def poison_features(X_tr, y_tr, pct):
    """Shuffle the order of features in `pct`% of training rows."""
    X_p = X_tr.copy()
    y_p = y_tr.copy()
    n_poison = max(1, int(len(y_p) * pct / 100))
    idx = np.random.choice(len(y_p), n_poison, replace=False)
    for i in idx:
        np.random.shuffle(X_p[i])   # in-place feature shuffle
    return X_p, y_p, n_poison


pcts_feat  = [50, 40, 30, 20, 1]
accs_feat  = []
accs_label = []

tcomp = Table(title="[bold white]Label-Flip vs Feature-Shuffle Accuracy[/bold white]",
              box=box.ROUNDED, show_lines=True, padding=(0, 2))
tcomp.add_column("Poison %",            style="bold white",        justify="center")
tcomp.add_column("Label-flip Acc",      style="bold bright_red",   justify="center")
tcomp.add_column("Diff vs Clean",       style="bold bright_red",   justify="center")
tcomp.add_column("Feature-shuffle Acc", style="bold bright_yellow", justify="center")
tcomp.add_column("Diff vs Clean",       style="bold bright_yellow", justify="center")
tcomp.add_column("Lbl vs Feat Diff",    style="bold bright_cyan",  justify="center")

for pct in pcts_feat:
    np.random.seed(42)
    Xpl, ypl, _ = poison_labels(X_train, y_train, pct)
    ml, _        = train_model(Xpl, ypl, verbose=False)
    al           = evaluate(ml, X_test, y_test)

    np.random.seed(42)
    Xpf, ypf, _ = poison_features(X_train, y_train, pct)
    mf, _        = train_model(Xpf, ypf, verbose=False)
    af           = evaluate(mf, X_test, y_test)

    accs_label.append(al)
    accs_feat.append(af)
    dl   = al - acc_clean
    df   = af - acc_clean
    diff = af - al
    dl_str   = f"{dl:+.2f}%"
    df_str   = f"{df:+.2f}%"
    diff_str = f"{diff:+.2f}%"
    tcomp.add_row(f"{pct}%", f"{al:.2f}%", dl_str, f"{af:.2f}%", df_str, diff_str)

console.print(tcomp)

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
ax.set_title('Steps 12-14 - Label-flip vs. Feature-shuffle Poisoning', fontsize=14, fontweight='bold', color='#00d4ff')
ax.legend(fontsize=11)
ax.grid(axis='y')
plt.tight_layout()
plt.savefig('picture3_labelflip_vs_featureshuffle.png', dpi=150, bbox_inches='tight')
plt.close()
console.print("  [dim]Saved: picture3_labelflip_vs_featureshuffle.png[/dim]")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
console.print()
console.print(Panel("[bold bright_white]FINAL SUMMARY  --  VARIANT 9[/bold bright_white]",
                    style="bold magenta", box=box.DOUBLE_EDGE))

summary = Table(box=box.ROUNDED, show_lines=True, padding=(0, 2),
                title="[bold white]All Experiments[/bold white]")
summary.add_column("Experiment",   style="bold white",        justify="left")
summary.add_column("Accuracy",     style="bold bright_green", justify="right")
summary.add_column("vs Clean",     style="bold bright_red",   justify="right")

def delta(a):
    d = a - acc_clean
    return f"{d:+.2f}%" if d != 0 else "[green]baseline[/green]"

summary.add_row("Clean model",             f"{acc_clean:.2f}%", "[green]baseline[/green]")
summary.add_row("Label-flip 50% (N=50%)", f"{acc_p50:.2f}%",  delta(acc_p50))
for lbl, acc in zip(labels_b2[1:], accs_b2[1:]):   # skip 50% already added above
    summary.add_row(f"Label-flip {lbl}", f"{acc:.2f}%", delta(acc))
summary.add_row("Label-flip 1%",          f"{acc_1pct:.2f}%", delta(acc_1pct))
summary.add_row("Label-flip 1 row",       f"{acc_1row:.2f}%", delta(acc_1row))

console.print(summary)

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
console.print("  [dim]Saved: picture4_training_loss.png[/dim]")

console.print()
console.print(Panel("[bold bright_green]  ALL STEPS COMPLETE!  Check the saved PNG files for plots.  [/bold bright_green]",
                    style="bold green", box=box.DOUBLE_EDGE))
console.print()
