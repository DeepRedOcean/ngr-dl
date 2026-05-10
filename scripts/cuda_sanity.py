"""CUDA + PyTorch sanity check.

Trains a small MLP to regress y = sin(x) + Gaussian noise, x in [-pi, pi].
Confirms end-to-end training works on GPU within this project's environment.
This file is the canonical template for every later training script.
"""

import math
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---- Reproducibility ------------------------------------------------------
# Fixing the seed so re-runs produce the same numbers. Useful for debugging.
torch.manual_seed(42)

# ---- Device selection -----------------------------------------------------
# Explicit. Every tensor we touch must live on this device to train on GPU.
# .to(device) returns a NEW tensor on that device (it's not in-place).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Synthetic data: y = sin(x) + small Gaussian noise --------------------
# 10,000 samples, x uniformly in [-pi, pi], y = sin(x) + N(0, 0.1).
# Shape convention: (N, 1) so that one row = one sample, one column = the
# scalar feature. nn.Linear expects (batch, features).
N = 10_000
x = torch.empty(N, 1).uniform_(-math.pi, math.pi)
y = torch.sin(x) + 0.1 * torch.randn(N, 1)

# TensorDataset wraps (x, y) so DataLoader can iterate in batches.
# DataLoader handles shuffling and batching. shuffle=True at every epoch
# matters: it prevents the model from over-fitting to the order of samples.
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)


# ---- Model ----------------------------------------------------------------
# nn.Module is THE standard base class. Subclass it and define forward().
# Backward is derived automatically by autograd from the forward computation.
# nn.Sequential just chains layers; it's syntactic sugar for forward().
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()  # always call this first
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 1 -> 64
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 64 -> 64
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # 64 -> 1, no activation (regression)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Instantiate and move to device. .to() handles all the model's parameters.
model = SimpleMLP().to(device)

# ---- Loss + optimizer -----------------------------------------------------
# MSE = mean squared error, the standard L2 regression loss.
loss_fn = nn.MSELoss()
# Adam: adaptive optimizer, sane default for almost everything.
# lr=1e-3 is the canonical first guess.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5

# wandb init
wandb.init(
    project="ngr-dl",
    name="cuda-sanity",
    config={
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": EPOCHS,
        "model": "MLP-1-64-64-1",
        "precision": "bf16-autocast",
    },
)

# ---- Initial loss (before any training) -----------------------------------
# Sanity reference. torch.no_grad() disables autograd bookkeeping during eval
# (saves memory, faster).
with torch.no_grad():
    init_loss = loss_fn(model(x.to(device)), y.to(device)).item()
print(f"Initial loss: {init_loss:.4f}")

# wandb utilities:
global_step = 0

# ---- Training loop --------------------------------------------------------
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n_batches = 0

    model.train()  # switches dropout/batchnorm to train mode (no-op here, habit)
    for x_batch, y_batch in loader:
        # Move this batch to GPU.
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass with autocast bf16
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

        # Backward pass + parameter update.
        # zero_grad clears stale gradients from the previous step (otherwise
        # they accumulate). loss.backward() computes gradients via autograd.
        # optimizer.step() updates parameters using those gradients.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)
        global_step += 1

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    print(f"Epoch {epoch + 1}/{EPOCHS} - avg loss: {avg_loss:.4f}")
    wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch}, step=global_step)



# ---- Final loss -----------------------------------------------------------
model.eval()
with torch.no_grad():
    final_loss = loss_fn(model(x.to(device)), y.to(device)).item()
print(f"Final loss: {final_loss:.4f}")
print(f"Loss reduction: {init_loss:.4f} -> {final_loss:.4f}")


wandb.finish()