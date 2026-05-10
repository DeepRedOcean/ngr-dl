# Project Journal



## Day 1 — 2026-05-09
- Repo, skeleton, CI green.
- Notes:

## Day 2 — 2026-05-10
- CUDA verified on RTX 3080 (cap 8.6, ~10GB).
- Pinned project to Python 3.13 for torch CUDA wheel compat.
- bf16 autocast working in training loop.
- W&B logging integrated. First run: https://wandb.ai/alexchoj/ngr-dl/runs/<id>
- Lessons: PyTorch idioms (nn.Module + autograd) vs manual implementations; ordering matters in scripts (EPOCHS before wandb.init); global_step lives outside epoch loop.