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

## Day 3 — 2026-07-31
- Back after 2 months. Pivoted to problem-first exploration instead of docs-first.
- Loaded Roman TDS truth image (SCA 17, F184 band, 4088x4088). Diffraction spikes visible on brightest source.
- First real Roman pixels on screen. Return-session win.

## Day 3 continued — 2026-07-31
- Return-session win: real Roman pixels on screen (first_peek.py).
- DAOStarFinder for point sources (2882 detections tuned via absolute threshold — noiseless truth image breaks the usual 5-sigma default).
- photutils.segmentation for both point + extended sources (1655 segments — catches galaxies DAOStarFinder rejected).
- Added IMPROVEMENTS.md as a living backlog.
- Added scikit-image as dep (photutils deblending needs it).
- Learned: detection threshold has to match the noise floor of the specific image; there's no universal default.

## Day 6 — 2026-08-07

- Expanded typer training data: downloaded 4 additional (image + catalog) pairs from IRSA Roman TDS preview (visits 10307/18, 10692/17-18, 11077/17 — some visits 404'd because not every visit has both SCAs in preview). Total: 5 SCAs, 8739 stamps, 721 stars (6.2x v0).
- Refactored extract_stamps.py for multi-file input (auto-discovers pairs, tracks visit/SCA per stamp).
- Trained typer v1 on expanded data (same architecture, same hyperparams). Results:
  - Test accuracy 0.958 -> 0.995
  - Star recall 0.583 -> 0.959   (+0.376!)
  - Star F1 0.683 -> 0.972
  - Missed stars: 10/24 (v0) -> 6/145 (v1)
- Data-scale hypothesis empirically confirmed: architecture is capable, was starved of examples.
- Misclassification analysis: last 6 missed stars all at faintest mag bin (4.7-5.4). Two false-positive galaxies now borderline confidence (P=0.64, 0.82) vs v0's confident errors (P=0.94) — model well-calibrated in its errors.
- Added standing workflow: fetch date at session start, always include venv activation with cd commands, auto-update JOURNAL, auto-add IMPROVEMENTS items.

## Day 5 — 2026-08-06

- Built the first real ML model on Roman data: Stage 1 typer CNN (binary star vs galaxy).
- Pipeline: extract_stamps.py -> stamps_10307_17.npz (1536 stamps, 1419 galaxy / 117 star, 12:1 imbalance, transients dropped) -> train_typer.py (TinyTyper, 25k params, class-weighted CE, bf16 autocast, wandb) -> inspect_misclassified.py.
- Test set: 309 stamps. Accuracy 95.8%, but real metric is star recall = 58.3% (14/24). Star precision 82.4%, F1 0.68. Galaxy precision 96.6%, recall 98.9%.
- Interesting training pathology: at epoch 14-16 val_loss spiked to 2.5 (from 0.24) while train_loss kept falling — BatchNorm running-stat instability on tiny val set. Early stopping saved epoch-12 checkpoint.
- Misclassification analysis: missed stars cluster at faint end (mag 2-5), correctly-classified stars span full mag range down to mag -4. The 2 false-positive galaxies are highly compact galaxies indistinguishable from stars without color info.
- GPU actually earned its keep for the first time — training completed in a few seconds.
- Cascade architecture validated OPERATIONALLY, not just conceptually. Working typer exists.

## Day 4 — 2026-08-06
- Downloaded Roman TDS truth INDEX catalog (per-image ground truth, 14925 objects on this SCA: 14694 galaxies + 165 stars + 66 transients).
- cross_match.py: pixel-space cross-match, purity/completeness with per-type + per-mag breakdown.
  - Purity 0.962, completeness 0.107 at threshold=100. Star completeness 73% vs galaxy 10% — the depth cliff.
- threshold_sweep.py: 13 thresholds from 0.5 to 1000.
  - Non-monotonic completeness peak around threshold 1-2 (0.555 total).
  - Ceiling at ~55% for simple segmentation — motivates ML-based detection.
- per_type_threshold.py: fine-grained per-type threshold search.
  - Peaks: galaxy=1.3 (0.564), star=15 (0.970), transient=3 (0.303).
  - Specialist-cascade aggregate gain over single-global is only +19 objects (+0.001) because galaxies dominate truth 98.5% — but per-class the win is real (stars 0.87→0.97, transients 0.27→0.30).
  - Real specialist benefit: cleaner purity per branch (0.77 at t=1.3 vs 0.96 at t=15). Each downstream classifier gets a cleaner input.
- Added architectural finding + segmentation-ceiling insight to IMPROVEMENTS.md.
- No-GPU-here explanation: classical detection is CPU-only by design; GPU will earn its keep in Phase 2 SSL training.
