"""Train the Stage 1 typer CNN: binary star vs galaxy classifier.

Loads postage stamps extracted by extract_stamps.py, trains a small CNN with
class-weighted CrossEntropy (12:1 imbalance), evaluates on a stratified test
split, and logs everything to Weights & Biases.

Architecture: 3 conv blocks (16/32/64 channels) + global average pool +
small linear head. ~25k parameters — deliberately tiny to reduce overfitting
on our small dataset (~1500 stamps total).

Preprocessing: asinh transform to compress dynamic range (bright stars vs.
faint galaxies span millions of counts), then per-stamp z-score normalize.

Evaluation: accuracy, per-class precision/recall/F1, confusion matrix.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import wandb

# ---- Config ---------------------------------------------------------------
STAMPS_PATH = Path("data/stamps/stamps_all.npz")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 6

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load data + stratified split -----------------------------------------
data = np.load(STAMPS_PATH, allow_pickle=True)
stamps = data["stamps"]  # (N, 64, 64) float32
labels = data["labels"]  # (N,) int64
class_names = list(data["class_names"])  # e.g. ["galaxy", "star"]
print(
    f"Loaded {len(stamps)} stamps, class distribution: "
    f"{np.bincount(labels).tolist()} = {class_names}"
)


def stratified_split(labels, train_frac=0.6, val_frac=0.2, seed=42):
    """Split indices into train/val/test with per-class stratification."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train : n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val :])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


train_idx, val_idx, test_idx = stratified_split(labels, seed=SEED)
print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    counts = np.bincount(labels[idx], minlength=len(class_names))
    print(f"  {name:<5}  {counts.tolist()}  ({class_names})")


# ---- Dataset --------------------------------------------------------------
class StampDataset(Dataset):
    """Per-stamp asinh + z-score normalization at __getitem__ time.

    We do it lazily rather than precomputing so we can add augmentation
    later without re-precomputing the whole array.
    """

    def __init__(self, stamps: np.ndarray, labels: np.ndarray):
        self.stamps = stamps
        self.labels = labels

    def __len__(self):
        return len(self.stamps)

    def __getitem__(self, i):
        stamp = self.stamps[i].astype(np.float32)
        # asinh compresses the ~million-fold dynamic range.
        stamp = np.arcsinh(stamp)
        # Per-stamp z-score. eps guards against constant-image edge cases.
        mean, std = stamp.mean(), stamp.std()
        stamp = (stamp - mean) / (std + 1e-6)
        # Add channel dim: (64, 64) -> (1, 64, 64)
        return torch.from_numpy(stamp).unsqueeze(0), int(self.labels[i])


train_ds = StampDataset(stamps[train_idx], labels[train_idx])
val_ds = StampDataset(stamps[val_idx], labels[val_idx])
test_ds = StampDataset(stamps[test_idx], labels[test_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)


# ---- Model ----------------------------------------------------------------
class TinyTyper(nn.Module):
    """Small CNN for 64x64 monochromatic postage stamps -> 2-class logits."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 16, 64x64 -> 32x32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 16 -> 32, 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 32 -> 64, 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # GAP collapses spatial dims: 8x8x64 -> 64. Small head prevents
        # overfitting on our tiny dataset.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


model = TinyTyper(n_classes=len(class_names)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: TinyTyper, {n_params} parameters")

# ---- Loss + optimizer -----------------------------------------------------
# Class weights computed from TRAIN split only (not full dataset — avoid
# leaking val/test info). Inverse-frequency weighting.
train_counts = np.bincount(labels[train_idx], minlength=len(class_names))
class_weights = torch.tensor(
    train_counts.sum() / (len(class_names) * train_counts),
    dtype=torch.float32,
).to(device)
print(f"Class weights (from train): {class_weights.tolist()}")

loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ---- W&B ------------------------------------------------------------------
wandb.init(
    project="ngr-dl",
    name="stage1-typer-v1-multifile",
    config={
        "arch": "TinyTyper",
        "n_params": n_params,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "seed": SEED,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "class_names": class_names,
        "class_weights": class_weights.tolist(),
        "precision": "bf16-autocast",
    },
)


# ---- Train / eval helpers -------------------------------------------------
def run_epoch(loader, train: bool):
    """One pass over a loader. Returns (avg_loss, accuracy, y_true, y_pred)."""
    model.train() if train else model.eval()
    total_loss = 0.0
    n_seen = 0
    all_true, all_pred = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for stamp, label in loader:
            stamp = stamp.to(device)
            label = label.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(stamp)
                loss = loss_fn(logits, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(label)
            n_seen += len(label)
            pred = logits.argmax(dim=1)
            all_true.extend(label.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    avg_loss = total_loss / n_seen
    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    acc = float((y_true == y_pred).mean())
    return avg_loss, acc, y_true, y_pred


# ---- Training loop --------------------------------------------------------
best_val_loss = float("inf")
epochs_no_improve = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_acc, _, _ = run_epoch(val_loader, train=False)

    wandb.log(
        {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
        }
    )

    print(
        f"Epoch {epoch:>3}/{EPOCHS}  "
        f"train_loss={train_loss:.4f} acc={train_acc:.3f}  |  "
        f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no val improvement in {EARLY_STOP_PATIENCE} epochs)."
            )
            break

# Restore best-val checkpoint before test eval.
if best_state is not None:
    model.load_state_dict(best_state)
    ckpt_path = CHECKPOINTS_DIR / "stage1_typer_v1.pt"
    torch.save(best_state, ckpt_path)
    print(f"\nSaved best checkpoint: {ckpt_path}")

# ---- Test evaluation ------------------------------------------------------
test_loss, test_acc, y_true, y_pred = run_epoch(test_loader, train=False)
print()
print("=" * 60)
print("Test set evaluation")
print("=" * 60)
print(f"  Test loss:     {test_loss:.4f}")
print(f"  Test accuracy: {test_acc:.3f}")

# Per-class precision / recall / F1 (no sklearn — small manual computation).
print("\nPer-class metrics:")
for cls_idx, cls_name in enumerate(class_names):
    tp = int(((y_pred == cls_idx) & (y_true == cls_idx)).sum())
    fp = int(((y_pred == cls_idx) & (y_true != cls_idx)).sum())
    fn = int(((y_pred != cls_idx) & (y_true == cls_idx)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(
        f"  {cls_name:<10}  tp={tp:>4}  fp={fp:>4}  fn={fn:>4}  "
        f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}"
    )
    wandb.log(
        {
            f"test/{cls_name}/precision": precision,
            f"test/{cls_name}/recall": recall,
            f"test/{cls_name}/f1": f1,
        }
    )

# Confusion matrix.
n_classes = len(class_names)
cm = np.zeros((n_classes, n_classes), dtype=int)
for t, p in zip(y_true, y_pred, strict=True):
    cm[t, p] += 1

print("\nConfusion matrix (rows = truth, cols = predicted):")
header = "true \\ pred  " + "  ".join(f"{n:>8}" for n in class_names)
print(header)
for i, name in enumerate(class_names):
    row = f"{name:<12}  " + "  ".join(f"{cm[i, j]:>8}" for j in range(n_classes))
    print(row)

wandb.log(
    {
        "test/loss": test_loss,
        "test/accuracy": test_acc,
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=class_names,
        ),
    }
)

# ---- Confusion matrix figure ----------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Truth")
ax.set_title(f"Test confusion matrix — accuracy {test_acc:.3f}")
for i in range(n_classes):
    for j in range(n_classes):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "typer_confusion.png", dpi=140)
print(f"\nSaved: {FIGURES_DIR / 'typer_confusion.png'}")

wandb.finish()
print("\nDone.")
