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

## Day 4 — 2026-07-31 (second half of the day)
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
