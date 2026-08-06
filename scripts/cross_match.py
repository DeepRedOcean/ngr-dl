"""Cross-match detected sources against the OpenUniverse truth catalog.

Measures COMPLETENESS (fraction of real objects we found) and PURITY
(fraction of our detections that are real) — the two canonical evaluation
metrics for any source-detection pipeline.

Data:
  - Image:       data/openuniverse_preview/Roman_TDS_truth_F184_10307_17.fits.gz
  - Truth cat:   data/openuniverse_preview/Roman_TDS_index_F184_10307_17.txt
    ASCII commented-header table with columns:
      object_id, ra, dec, x, y, realized_flux, flux, mag, obj_type

Method:
  1. Re-run segmentation detection to get (x_det, y_det) for each source.
  2. Load truth catalog; filter to objects actually on the SCA (0 <= x < 4088).
  3. Cross-match in pixel space (both have (x, y)) using scipy cKDTree,
     tolerance ~5 px (~0.5" at Roman's ~0.11"/px plate scale).
  4. Compute completeness + purity, broken down by object type and magnitude.
  5. Visualize matches (green), false positives (red), missed truth (yellow).

Why pixel-space cross-match: OpenUniverse's per-image truth catalog helpfully
carries pixel coordinates already. For non-OpenUniverse catalogs (or
multi-epoch matching across SCAs), we'd need proper sky-coord matching via
astropy WCS + SkyCoord.match_to_catalog_sky. That's queued in IMPROVEMENTS.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import photutils
from astropy.convolution import convolve
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from photutils.segmentation import SourceCatalog, SourceFinder, make_2dgaussian_kernel
from scipy.spatial import cKDTree

photutils.future_column_names = True

# ---- Paths & config -------------------------------------------------------
DATA_DIR = Path("data/openuniverse_preview")
IMAGE_PATH = DATA_DIR / "Roman_TDS_truth_F184_10307_17.fits.gz"
TRUTH_PATH = DATA_DIR / "Roman_TDS_index_F184_10307_17.txt"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

MATCH_TOL_PX = 5.0  # pixels; Roman plate scale ~0.11"/px so ~5px ~= 0.55"

# ---- Load image + re-run detection ----------------------------------------
with fits.open(IMAGE_PATH) as hdul:
    image = hdul[0].data.astype(np.float32)

_, median, _ = sigma_clipped_stats(image, sigma=3.0)
kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
convolved = convolve(image - median, kernel)
finder = SourceFinder(n_pixels=10, progress_bar=False)
segment_map = finder(convolved, threshold=100.0)

catalog = SourceCatalog(image - median, segment_map)
det = catalog.to_table(columns=["label", "x_centroid", "y_centroid", "area", "segment_flux"])
print(f"Detections re-computed: {len(det)}")

# ---- Load truth catalog ---------------------------------------------------
# The file starts with a '#'-commented header line naming the columns —
# astropy's 'ascii.commented_header' format handles this natively.
truth = Table.read(TRUTH_PATH, format="ascii.commented_header")
print(f"Truth catalog rows (all): {len(truth)}")
print(f"Truth columns:            {truth.colnames}")

types_present = sorted(set(truth["obj_type"]))
print(f"Object types in truth:    {types_present}")

# Filter to objects that actually land on this SCA. The truth catalog
# includes objects slightly outside the chip because the simulation
# considers a wider region (border effects, dithering, etc.).
Ny, Nx = image.shape
on_sca_mask = (truth["x"] >= 0) & (truth["x"] < Nx) & (truth["y"] >= 0) & (truth["y"] < Ny)
truth_on = truth[on_sca_mask]
print(f"Truth objects ON SCA:     {len(truth_on)}  (of {len(truth)} in file)")

# Print the magnitude range so we know what we're dealing with. OpenUniverse
# "mag" column is a simulation-internal normalization, not standard AB.
mag_arr = np.asarray(truth_on["mag"], dtype=float)
print(f"Truth mag range:          {mag_arr.min():.2f} .. {mag_arr.max():.2f}")

# ---- Pixel-space cross-match ----------------------------------------------
# For each detection, find the nearest truth object by pixel distance.
# scipy cKDTree makes this fast even for 10k+ objects.
truth_xy = np.column_stack([np.asarray(truth_on["x"]), np.asarray(truth_on["y"])])
det_xy = np.column_stack([np.asarray(det["x_centroid"]), np.asarray(det["y_centroid"])])

tree = cKDTree(truth_xy)
dist, idx = tree.query(det_xy, k=1)

matched_det_mask = dist < MATCH_TOL_PX
det["match_dist_px"] = dist
det["truth_idx"] = idx  # index into truth_on

n_detections = len(det)
n_matched_det = int(matched_det_mask.sum())
n_truth = len(truth_on)

# For each matched detection, record which truth object it hit. A truth
# object is "found" if any detection matched it (a real object might match
# zero, one, or several detections — we count it as "found" once).
matched_truth_set = set(int(i) for i in det["truth_idx"][matched_det_mask])
n_truth_found = len(matched_truth_set)

purity = n_matched_det / n_detections
completeness = n_truth_found / n_truth

# ---- Report ---------------------------------------------------------------
print()
print("=" * 60)
print(f"Cross-match ({MATCH_TOL_PX}px tolerance)")
print("=" * 60)
print(f"  Detections:              {n_detections}")
print(f"  Truth objects on SCA:    {n_truth}")
print(f"  Matched detections:      {n_matched_det}")
print(f"  Distinct truth found:    {n_truth_found}")
print(f"  Purity   (matched/det):     {purity:.3f}")
print(f"  Completeness (found/truth): {completeness:.3f}")

# Break down completeness by object type: how well do we find stars vs galaxies?
print("\nCompleteness by object type:")
obj_types = np.asarray(truth_on["obj_type"])
for t in sorted(set(obj_types.tolist())):
    total = int((obj_types == t).sum())
    truth_indices_of_type = set(np.where(obj_types == t)[0].tolist())
    found = len(matched_truth_set & truth_indices_of_type)
    frac = found / total if total > 0 else 0.0
    print(f"  {t:<10}  {found:>6}/{total:<6}  ({frac:.3f})")

# Break down by magnitude using quintile bins — always sensible regardless
# of the mag column's absolute scale.
print("\nCompleteness by magnitude quintile (fainter = larger mag):")
bin_edges = np.percentile(mag_arr, [0, 20, 40, 60, 80, 100])
for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    if i == len(bin_edges) - 2:
        in_bin = (mag_arr >= lo) & (mag_arr <= hi)
    else:
        in_bin = (mag_arr >= lo) & (mag_arr < hi)
    total = int(in_bin.sum())
    if total == 0:
        continue
    truth_indices_in_bin = set(np.where(in_bin)[0].tolist())
    found = len(matched_truth_set & truth_indices_in_bin)
    frac = found / total
    print(f"  mag {lo:7.3f} .. {hi:7.3f}   {found:>6}/{total:<6}  ({frac:.3f})")

# ---- Visualization: full-frame overlay ------------------------------------
norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=AsinhStretch())

missed_truth_indices = np.array(
    [i for i in range(n_truth) if i not in matched_truth_set], dtype=int
)
missed_x = np.asarray(truth_on["x"])[missed_truth_indices]
missed_y = np.asarray(truth_on["y"])[missed_truth_indices]

fig, ax = plt.subplots(figsize=(11, 11))
ax.imshow(image, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
ax.scatter(
    det["x_centroid"][matched_det_mask],
    det["y_centroid"][matched_det_mask],
    s=8,
    facecolors="none",
    edgecolors="lime",
    linewidths=0.4,
    label=f"matched detection ({n_matched_det})",
)
ax.scatter(
    det["x_centroid"][~matched_det_mask],
    det["y_centroid"][~matched_det_mask],
    s=8,
    facecolors="none",
    edgecolors="red",
    linewidths=0.4,
    label=f"false positive ({n_detections - n_matched_det})",
)
ax.scatter(
    missed_x,
    missed_y,
    s=12,
    marker="+",
    c="gold",
    linewidths=0.4,
    label=f"missed truth ({len(missed_truth_indices)})",
)
ax.legend(loc="lower right", framealpha=0.85)
ax.set_title(
    f"Cross-match — completeness={completeness:.3f}, purity={purity:.3f}, " f"tol={MATCH_TOL_PX}px"
)
ax.set_xlabel("pixel x")
ax.set_ylabel("pixel y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "cross_match.png", dpi=140)
print(f"\nSaved: {FIGURES_DIR / 'cross_match.png'}")

# ---- Zoom around brightest matched source ---------------------------------
if n_matched_det > 0:
    # Sort by detection segment flux, pick the brightest matched one.
    matched_det = det[matched_det_mask]
    matched_det.sort("segment_flux", reverse=True)
    brightest = matched_det[0]
    bx, by = float(brightest["x_centroid"]), float(brightest["y_centroid"])

    half = 256
    y0, y1 = max(0, int(by - half)), min(Ny, int(by + half))
    x0, x1 = max(0, int(bx - half)), min(Nx, int(bx + half))
    zoom = image[y0:y1, x0:x1]
    zoom_norm = ImageNormalize(zoom, interval=ZScaleInterval(), stretch=AsinhStretch())

    in_zoom_det = (
        (det["x_centroid"] >= x0)
        & (det["x_centroid"] < x1)
        & (det["y_centroid"] >= y0)
        & (det["y_centroid"] < y1)
    )
    in_zoom_missed = (missed_x >= x0) & (missed_x < x1) & (missed_y >= y0) & (missed_y < y1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        zoom,
        origin="lower",
        cmap="gray",
        norm=zoom_norm,
        interpolation="nearest",
        extent=[x0, x1, y0, y1],
    )
    ax.scatter(
        det["x_centroid"][in_zoom_det & matched_det_mask],
        det["y_centroid"][in_zoom_det & matched_det_mask],
        s=60,
        facecolors="none",
        edgecolors="lime",
        linewidths=1.2,
        label="matched",
    )
    ax.scatter(
        det["x_centroid"][in_zoom_det & ~matched_det_mask],
        det["y_centroid"][in_zoom_det & ~matched_det_mask],
        s=60,
        facecolors="none",
        edgecolors="red",
        linewidths=1.2,
        label="false pos",
    )
    ax.scatter(
        missed_x[in_zoom_missed],
        missed_y[in_zoom_missed],
        s=80,
        marker="+",
        c="gold",
        linewidths=1.5,
        label="missed",
    )
    ax.legend(loc="lower right", framealpha=0.85)
    ax.set_title("Zoom around brightest matched source")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_match_zoom.png", dpi=140)
    print(f"Saved: {FIGURES_DIR / 'cross_match_zoom.png'}")

print("\nDone.")
