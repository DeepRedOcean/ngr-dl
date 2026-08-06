"""Per-type threshold analysis: validate "each specialist has its own
optimal detection threshold" hypothesis from the previous sweep.

Fine-grained threshold values targeted at each object type's expected peak
(peaks estimated from the coarse threshold_sweep.py results):
  - Galaxy    : 0.9, 1.0, 1.1, 1.2, 1.3  (coarse peak was ~1)
  - Star      : 15, 18, 20, 22, 25       (coarse peak was ~20)
  - Transient : 2, 3, 4, 5, 6            (coarse peak was ~3-5)

For each threshold in the union, run detection + cross-match, record
per-type completeness. Then:
  - Print per-type peak thresholds and completeness values.
  - Compare "best single global threshold" vs "best per-type thresholds":
    quantify the completeness gain from a specialist-cascade architecture.
  - Plot per-type completeness curves with each type's peak marked.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import photutils
from astropy.convolution import convolve
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.segmentation import SourceCatalog, SourceFinder, make_2dgaussian_kernel
from scipy.spatial import cKDTree

photutils.future_column_names = True

DATA_DIR = Path("data/openuniverse_preview")
IMAGE_PATH = DATA_DIR / "Roman_TDS_truth_F184_10307_17.fits.gz"
TRUTH_PATH = DATA_DIR / "Roman_TDS_index_F184_10307_17.txt"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Per-type target thresholds; the actual sweep uses the sorted union so we
# don't repeat detection runs unnecessarily.
TARGET_PER_TYPE = {
    "galaxy": [0.9, 1.0, 1.1, 1.2, 1.3],
    "star": [15, 18, 20, 22, 25],
    "transient": [2, 3, 4, 5, 6],
}
THRESHOLDS = sorted({t for lst in TARGET_PER_TYPE.values() for t in lst})
MATCH_TOL_PX = 5.0

TYPE_ORDER = ["galaxy", "star", "transient"]
TYPE_COLORS = {"galaxy": "tab:blue", "star": "tab:orange", "transient": "tab:green"}

# ---- Load image, truth, prep KDTree ---------------------------------------
with fits.open(IMAGE_PATH) as hdul:
    image = hdul[0].data.astype(np.float32)

_, median, _ = sigma_clipped_stats(image, sigma=3.0)
Ny, Nx = image.shape

truth = Table.read(TRUTH_PATH, format="ascii.commented_header")
on_sca = (truth["x"] >= 0) & (truth["x"] < Nx) & (truth["y"] >= 0) & (truth["y"] < Ny)
truth_on = truth[on_sca]
truth_xy = np.column_stack([np.asarray(truth_on["x"]), np.asarray(truth_on["y"])])
truth_tree = cKDTree(truth_xy)
obj_types = np.asarray(truth_on["obj_type"])
n_truth = len(truth_on)

per_type_totals = {t: int((obj_types == t).sum()) for t in TYPE_ORDER}
print(f"Truth on SCA: {n_truth}    Per-type totals: {per_type_totals}")

kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
convolved = convolve(image - median, kernel)

# ---- Sweep -----------------------------------------------------------------
results = []
header_fmt = "{:>7}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}"
row_fmt = "{:>7}  {:>7}  {:>9.3f}  {:>9.3f}  {:>9.3f}  {:>9.3f}  {:>6.1f}s"
print()
print(
    header_fmt.format("thresh", "n_det", "purity", "gal_comp", "star_comp", "tra_comp", "elapsed")
)
print("-" * 72)

for i, threshold in enumerate(THRESHOLDS, 1):
    print(f"  [{i}/{len(THRESHOLDS)}] running threshold={threshold} ...", flush=True, end="")
    t_start = time.time()

    finder = SourceFinder(n_pixels=10, progress_bar=False)
    segment_map = finder(convolved, threshold=float(threshold))

    if segment_map is None or segment_map.n_labels == 0:
        elapsed = time.time() - t_start
        results.append({"threshold": threshold, "n_det": 0, "purity": np.nan, "per_type": {}})
        print(
            f"\r{threshold:>7}  {'---':>7}  {'---':>9}  {'---':>9}  "
            f"{'---':>9}  {'---':>9}  {elapsed:>6.1f}s",
            flush=True,
        )
        continue

    catalog = SourceCatalog(image - median, segment_map)
    det = catalog.to_table(columns=["x_centroid", "y_centroid"])
    det_xy = np.column_stack([np.asarray(det["x_centroid"]), np.asarray(det["y_centroid"])])
    dist, idx = truth_tree.query(det_xy, k=1)
    matched = dist < MATCH_TOL_PX

    n_det = len(det)
    n_matched = int(matched.sum())
    matched_truth_set = set(int(i) for i in idx[matched])
    purity = n_matched / n_det if n_det > 0 else float("nan")

    per_type_frac = {}
    for t in TYPE_ORDER:
        truth_indices_of_type = set(np.where(obj_types == t)[0].tolist())
        found = len(matched_truth_set & truth_indices_of_type)
        total = per_type_totals[t]
        per_type_frac[t] = (found, total, found / total if total > 0 else 0.0)

    elapsed = time.time() - t_start
    results.append(
        {
            "threshold": threshold,
            "n_det": n_det,
            "purity": purity,
            "per_type": per_type_frac,
        }
    )
    print(
        "\r"
        + row_fmt.format(
            threshold,
            n_det,
            purity,
            per_type_frac["galaxy"][2],
            per_type_frac["star"][2],
            per_type_frac["transient"][2],
            elapsed,
        ),
        flush=True,
    )

# ---- Per-type peaks -------------------------------------------------------
print()
print("=" * 60)
print("Per-type peaks")
print("=" * 60)

peaks = {}
for t in TYPE_ORDER:
    best_frac = -1.0
    best_t = None
    best_found = 0
    for r in results:
        if not r["per_type"]:
            continue
        found, total, frac = r["per_type"][t]
        if frac > best_frac:
            best_frac = frac
            best_t = r["threshold"]
            best_found = found
    peaks[t] = (best_t, best_frac, best_found)
    print(
        f"  {t:<10}  best threshold {best_t:<5}  "
        f"found {best_found}/{per_type_totals[t]}  ({best_frac:.3f})"
    )

# ---- Single-global vs per-type comparison ---------------------------------
# For a fair comparison, "best single global" = threshold that maximizes the
# TOTAL number of objects found across all three types.
best_single_score = -1
best_single_t = None
best_single_per_type = None
for r in results:
    if not r["per_type"]:
        continue
    score = sum(r["per_type"][t][0] for t in TYPE_ORDER)
    if score > best_single_score:
        best_single_score = score
        best_single_t = r["threshold"]
        best_single_per_type = r["per_type"]

total_truth_all = sum(per_type_totals.values())
per_type_combined = sum(peaks[t][2] for t in TYPE_ORDER)

print()
print("=" * 60)
print("Single global threshold vs per-type thresholds")
print("=" * 60)
print(f"\nBest single global threshold: {best_single_t}")
for t in TYPE_ORDER:
    found, total, frac = best_single_per_type[t]
    print(f"  {t:<10}  {found}/{total}  ({frac:.3f})")
print(
    f"  {'TOTAL':<10}  {best_single_score}/{total_truth_all}  "
    f"({best_single_score/total_truth_all:.3f})"
)

print("\nBest per-type thresholds (specialist cascade upper bound):")
for t in TYPE_ORDER:
    print(
        f"  {t:<10}  threshold {peaks[t][0]:<5}  "
        f"{peaks[t][2]}/{per_type_totals[t]}  ({peaks[t][1]:.3f})"
    )
print(
    f"  {'TOTAL':<10}  {per_type_combined}/{total_truth_all}  "
    f"({per_type_combined/total_truth_all:.3f})"
)

gain_objects = per_type_combined - best_single_score
gain_frac = gain_objects / total_truth_all
print(f"\nGain from per-type thresholds: +{gain_objects} objects  ({gain_frac:+.3f} completeness)")
print("Note: this is a raw upper bound — a real cascade also needs a Stage 1")
print("typer to correctly route each detection to the right specialist.")

# ---- Plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

threshs = [r["threshold"] for r in results if r["per_type"]]
for t in TYPE_ORDER:
    fracs = [r["per_type"][t][2] for r in results if r["per_type"]]
    peak_t, peak_frac, _ = peaks[t]
    color = TYPE_COLORS[t]
    ax.plot(
        threshs,
        fracs,
        "o-",
        color=color,
        linewidth=2,
        markersize=7,
        label=f"{t} (peak {peak_frac:.3f} @ t={peak_t})",
    )
    ax.axvline(peak_t, color=color, linestyle="--", alpha=0.5)
    ax.scatter(
        [peak_t],
        [peak_frac],
        color=color,
        s=200,
        zorder=5,
        facecolors="none",
        edgecolors=color,
        linewidths=2,
    )

ax.set_xscale("log")
ax.set_xlabel("Detection threshold (counts, log scale)")
ax.set_ylabel("Per-type completeness")
ax.set_title(
    "Per-type completeness — testing the 'specialist thresholds' hypothesis\n"
    f"(dashed vertical = peak per type; per-type "
    f"{per_type_combined}/{total_truth_all}"
    f" = {per_type_combined/total_truth_all:.3f} vs "
    f"single-global {best_single_score/total_truth_all:.3f})"
)
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="center left")
ax.set_ylim(-0.02, 1.05)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "per_type_threshold.png", dpi=140)
print(f"\nSaved: {FIGURES_DIR / 'per_type_threshold.png'}")

print("\nDone.")
