"""Visualize misclassified test samples from the Stage 1 typer.

Reproduces the exact same stratified split as train_typer.py (seed=42),
loads the saved checkpoint, runs inference on the test set, and shows the
misclassified stamps in a grid — missed stars (false negatives) and
false-positive galaxies. Also shows a reference grid of correctly-classified
stars for visual comparison, plus a magnitude-distribution comparison to
test the "faint stars are what we miss" hypothesis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---- Config ---------------------------------------------------------------
STAMPS_PATH = Path("data/stamps/stamps_all.npz")
CKPT_PATH = Path("checkpoints/stage1_typer_v1.pt")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Model (must match train_typer.py architecture) -----------------------
class TinyTyper(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
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


# ---- Reproduce train_typer.py's stratified split --------------------------
def stratified_split(labels, train_frac=0.6, val_frac=0.2, seed=42):
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


# ---- Load data + split ----------------------------------------------------
data = np.load(STAMPS_PATH, allow_pickle=True)
stamps = data["stamps"]
labels = data["labels"]
mags = data["mag"]
class_names = list(data["class_names"])

_, _, test_idx = stratified_split(labels, seed=SEED)
test_stamps = stamps[test_idx]
test_labels = labels[test_idx]
test_mags = mags[test_idx]
print(f"Test set: {len(test_stamps)} stamps, classes = {class_names}")

# ---- Load checkpoint + run inference --------------------------------------
model = TinyTyper(n_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()


def normalize(stamp):
    s = np.arcsinh(stamp.astype(np.float32))
    return (s - s.mean()) / (s.std() + 1e-6)


test_input = np.stack([normalize(s) for s in test_stamps])
test_input_t = torch.from_numpy(test_input).unsqueeze(1).to(device)
with torch.no_grad():
    logits = model(test_input_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()

# ---- Identify categories --------------------------------------------------
STAR = class_names.index("star")
GALAXY = class_names.index("galaxy")

missed_stars = (test_labels == STAR) & (preds == GALAXY)  # false negatives
fp_galaxies = (test_labels == GALAXY) & (preds == STAR)  # false positives
correct_stars = (test_labels == STAR) & (preds == STAR)  # true positives

print(f"Missed stars (FN):        {int(missed_stars.sum())}")
print(f"False-positive galaxies:  {int(fp_galaxies.sum())}")
print(f"Correctly ID'd stars:     {int(correct_stars.sum())}")


# ---- Grid plot helper -----------------------------------------------------
def plot_grid(stamps_arr, mags_arr, star_probs, title, save_path, max_n=16):
    n = min(len(stamps_arr), max_n)
    if n == 0:
        print(f"(no stamps for '{title}')")
        return
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n):
        ax = axes[i]
        s = np.arcsinh(stamps_arr[i].astype(np.float32))
        vmin, vmax = np.percentile(s, [1, 99.5])
        ax.imshow(s, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"mag={mags_arr[i]:.2f}\nP(star)={star_probs[i]:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"Saved: {save_path}")


plot_grid(
    test_stamps[missed_stars],
    test_mags[missed_stars],
    probs[missed_stars, STAR],
    f"Missed stars — truth=STAR, predicted=GALAXY  ({int(missed_stars.sum())} stamps)",
    FIGURES_DIR / "typer_missed_stars.png",
)

plot_grid(
    test_stamps[fp_galaxies],
    test_mags[fp_galaxies],
    probs[fp_galaxies, STAR],
    f"False-positive galaxies — truth=GALAXY, predicted=STAR  ({int(fp_galaxies.sum())} stamps)",
    FIGURES_DIR / "typer_false_pos_galaxies.png",
)

plot_grid(
    test_stamps[correct_stars],
    test_mags[correct_stars],
    probs[correct_stars, STAR],
    f"Correctly identified stars — reference "
    f"({int(correct_stars.sum())} stamps, up to 16 shown)",
    FIGURES_DIR / "typer_correct_stars.png",
)

# ---- Mag distribution: missed vs correct stars ----------------------------
# Testing the hypothesis "faint stars are what we miss".
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(
    float(min(test_mags[test_labels == STAR])),
    float(max(test_mags[test_labels == STAR])),
    12,
)
if missed_stars.sum() > 0:
    ax.hist(
        test_mags[missed_stars],
        bins=bins,
        alpha=0.65,
        label=f"missed ({int(missed_stars.sum())})",
        color="tab:red",
    )
if correct_stars.sum() > 0:
    ax.hist(
        test_mags[correct_stars],
        bins=bins,
        alpha=0.65,
        label=f"correct ({int(correct_stars.sum())})",
        color="tab:green",
    )
ax.set_xlabel("truth catalog mag (simulation-internal; smaller = brighter)")
ax.set_ylabel("count")
ax.set_title("Missed vs correctly-classified stars — magnitude distribution")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "typer_star_mag_dist.png", dpi=140)
print(f"Saved: {FIGURES_DIR / 'typer_star_mag_dist.png'}")

print("\nDone.")
