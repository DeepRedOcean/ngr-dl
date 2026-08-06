"""Detection threshold sweep: completeness vs. purity curve.

Runs segmentation-based detection at a range of threshold values, cross-
matches each result against the truth catalog, and plots the classic
completeness / purity trade-off. This is the operating characteristic of
the detector — the astronomy analogue of an ROC curve.

Expected behavior as threshold rises:
  * PURITY goes up   (fewer false positives)
  * COMPLETENESS goes down (miss more faint sources)

The right operating point depends on downstream use:
  * Anomaly-hunting (catch weird things)      -> lower threshold, higher recall
  * Clean catalog (few spurious sources)      -> higher threshold, higher precision
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

# ---- Config ---------------------------------------------------------------
DATA_DIR = Path("data/openuniverse_preview")
IMAGE_PATH = DATA_DIR / "Roman_TDS_truth_F184_10307_17.fits.gz"
TRUTH_PATH = DATA_DIR / "Roman_TDS_index_F184_10307_17.txt"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Threshold values to sweep across, in raw image counts. Log-spaced.
# Lower bound = 1 count, roughly at the noise floor (sigma-clipped std ~1.36).
# Going below that (0.1, 0.5) makes segmentation's deblending step blow up
# because half the image becomes one connected blob — not worth the compute.
THRESHOLDS = [0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
MATCH_TOL_PX = 5.0  # ~0.5" at Roman's ~0.11"/px

# ---- Load image + truth once ----------------------------------------------
with fits.open(IMAGE_PATH) as hdul:
    image = hdul[0].data.astype(np.float32)

_, median, _ = sigma_clipped_stats(image, sigma=3.0)
Ny, Nx = image.shape

truth = Table.read(TRUTH_PATH, format="ascii.commented_header")
on_sca = (truth["x"] >= 0) & (truth["x"] < Nx) & (truth["y"] >= 0) & (truth["y"] < Ny)
truth_on = truth[on_sca]
truth_xy = np.column_stack([np.asarray(truth_on["x"]), np.asarray(truth_on["y"])])
truth_tree = cKDTree(truth_xy)  # build KDTree once, reuse across thresholds
obj_types = np.asarray(truth_on["obj_type"])
n_truth = len(truth_on)

print(f"Image {image.shape}, background median {median:.4g}")
print(f"Truth objects on SCA: {n_truth}")

# Convolve once with the matched-filter kernel — result is what we threshold.
kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
convolved = convolve(image - median, kernel)

# ---- Sweep ----------------------------------------------------------------
results = []

header_fmt = "{:>7}  {:>7}  {:>10}  {:>13}  {:>7}  {:>9}  {:>7}"
row_fmt = "{:>7}  {:>7}  {:>10}  {:>13}  {:>7.3f}  {:>9.3f}  {:>6.1f}s"
print()
print(
    header_fmt.format(
        "thresh",
        "n_det",
        "n_matched",
        "n_truth_found",
        "purity",
        "complete",
        "elapsed",
    )
)
print("-" * 76)

for i, threshold in enumerate(THRESHOLDS, 1):
    print(f"  [{i}/{len(THRESHOLDS)}] running threshold={threshold} ...", flush=True, end="")
    t_start = time.time()

    finder = SourceFinder(n_pixels=10, progress_bar=False)
    segment_map = finder(convolved, threshold=float(threshold))

    if segment_map is None or segment_map.n_labels == 0:
        elapsed = time.time() - t_start
        results.append(
            {
                "threshold": threshold,
                "n_det": 0,
                "n_matched": 0,
                "n_truth_found": 0,
                "purity": np.nan,
                "completeness": 0.0,
                "per_type": {},
            }
        )
        print(
            f"\r{threshold:>7}  {'---':>7}  {'---':>10}  {'---':>13}  "
            f"{'---':>7}  {'---':>9}  {elapsed:>6.1f}s",
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
    n_truth_found = len(matched_truth_set)

    purity = n_matched / n_det if n_det > 0 else float("nan")
    completeness = n_truth_found / n_truth

    per_type = {}
    for t in sorted(set(obj_types.tolist())):
        total = int((obj_types == t).sum())
        truth_indices_of_type = set(np.where(obj_types == t)[0].tolist())
        found = len(matched_truth_set & truth_indices_of_type)
        per_type[t] = (found, total, found / total if total > 0 else 0.0)

    elapsed = time.time() - t_start
    results.append(
        {
            "threshold": threshold,
            "n_det": n_det,
            "n_matched": n_matched,
            "n_truth_found": n_truth_found,
            "purity": purity,
            "completeness": completeness,
            "per_type": per_type,
        }
    )
    # \r overwrites the "running..." line so we get one clean row per threshold
    print(
        "\r"
        + row_fmt.format(
            threshold,
            n_det,
            n_matched,
            n_truth_found,
            purity,
            completeness,
            elapsed,
        ),
        flush=True,
    )

# ---- Per-type completeness table ------------------------------------------
print()
types_present = sorted({t for r in results for t in r["per_type"]})
header = f"{'thresh':>7}  " + "  ".join(f"{t:>12}" for t in types_present)
print("Completeness by object type:")
print(header)
print("-" * len(header))
for r in results:
    if not r["per_type"]:
        continue
    row = [f"{r['threshold']:>7}"]
    for t in types_present:
        _, _, frac = r["per_type"].get(t, (0, 0, float("nan")))
        row.append(f"{frac:>12.3f}")
    print("  ".join(row))

# ---- Plot 1: trade-off curve + individual metrics vs threshold ------------
valid = [r for r in results if not np.isnan(r["purity"])]
thresh_arr = np.array([r["threshold"] for r in valid])
purity_arr = np.array([r["purity"] for r in valid])
compl_arr = np.array([r["completeness"] for r in valid])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: completeness vs purity, one point per threshold, colored by threshold
sc = ax1.scatter(
    purity_arr,
    compl_arr,
    c=np.log10(thresh_arr),
    s=120,
    cmap="viridis",
    edgecolors="black",
    zorder=3,
)
ax1.plot(purity_arr, compl_arr, "-", color="gray", alpha=0.5, zorder=1)
for t, p, c in zip(thresh_arr, purity_arr, compl_arr, strict=True):
    ax1.annotate(f"{int(t)}", (p, c), xytext=(7, 7), textcoords="offset points", fontsize=10)
ax1.set_xlabel("Purity (matched detections / total detections)")
ax1.set_ylabel("Completeness (distinct truth found / truth on SCA)")
ax1.set_title("Detection trade-off — label = threshold in counts")
ax1.set_xlim(-0.02, 1.05)
ax1.set_ylim(-0.02, max(compl_arr.max() * 1.15, 0.02))
ax1.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax1, label="log10(threshold)")

# Right: purity and completeness as functions of threshold
ax2.plot(thresh_arr, purity_arr, "o-", color="tab:blue", label="purity")
ax2.plot(thresh_arr, compl_arr, "s-", color="tab:orange", label="completeness")
ax2.set_xscale("log")
ax2.set_xlabel("Detection threshold (counts, log scale)")
ax2.set_ylabel("Fraction")
ax2.set_title("Purity & completeness vs threshold")
ax2.set_ylim(-0.02, 1.05)
ax2.legend(loc="center right")
ax2.grid(True, alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "threshold_sweep.png", dpi=140)
print(f"\nSaved: {FIGURES_DIR / 'threshold_sweep.png'}")

# ---- Plot 2: per-type completeness ----------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
for t in types_present:
    ys = [r["per_type"][t][2] for r in valid if t in r["per_type"]]
    xs = [r["threshold"] for r in valid if t in r["per_type"]]
    ax.plot(xs, ys, "o-", label=t)
ax.set_xscale("log")
ax.set_xlabel("Detection threshold (counts, log scale)")
ax.set_ylabel("Completeness")
ax.set_title("Type-specific completeness vs threshold")
ax.set_ylim(-0.02, 1.05)
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "threshold_sweep_by_type.png", dpi=140)
print(f"Saved: {FIGURES_DIR / 'threshold_sweep_by_type.png'}")

print("\nDone.")
