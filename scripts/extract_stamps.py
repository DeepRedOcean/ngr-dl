"""Extract postage stamps + labels for Stage 1 typer training (multi-file v1).

Auto-discovers all Roman_TDS_truth_F184_{visit}_{sca}.fits.gz files in the
data directory, cross-matches each against its INDEX catalog, cuts 64x64
stamps around every matched detection, and concatenates everything into one
big stamps_all.npz.

Per stamp we save (visit, sca) source so we can do per-image evaluation
later — e.g. train on visits {A, B, C, D} and test on held-out visit E.

Output shape:
  stamps       : (N, 64, 64) float32
  labels       : (N,)        int64   (0 = galaxy, 1 = star)
  det_x        : (N,)        float32
  det_y        : (N,)        float32
  mag          : (N,)        float32
  visit        : (N,)        int32   (source visit ID)
  sca          : (N,)        int32   (source SCA number)
  class_names  : (2,)        str
"""

import re
import time
from pathlib import Path

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
STAMPS_DIR = Path("data/stamps")
STAMPS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = STAMPS_DIR / "stamps_all.npz"

STAMP_SIZE = 64
HALF = STAMP_SIZE // 2
MATCH_TOL_PX = 5.0
DETECTION_THRESHOLD = 100.0

CLASS_TO_INT = {"galaxy": 0, "star": 1}

# Regex to pull (visit, sca) from filenames like
#   Roman_TDS_truth_F184_10307_17.fits.gz
FILENAME_RE = re.compile(r"Roman_TDS_truth_F184_(\d+)_(\d+)\.fits\.gz$")


# ---- Discover (image, catalog) pairs on disk ------------------------------
def discover_pairs() -> list[tuple[Path, Path, int, int]]:
    """Find all matching (image, catalog, visit, sca) tuples in DATA_DIR."""
    pairs = []
    for img_path in sorted(DATA_DIR.glob("Roman_TDS_truth_F184_*.fits.gz")):
        m = FILENAME_RE.search(img_path.name)
        if m is None:
            continue
        visit, sca = int(m.group(1)), int(m.group(2))
        cat_path = DATA_DIR / f"Roman_TDS_index_F184_{visit}_{sca}.txt"
        if not cat_path.exists():
            print(f"  skip {img_path.name}: no matching catalog {cat_path.name}")
            continue
        pairs.append((img_path, cat_path, visit, sca))
    return pairs


pairs = discover_pairs()
print(f"Discovered {len(pairs)} (image, catalog) pairs")
if not pairs:
    raise SystemExit("No pairs found — download data first.")

# Shared kernel — recreate per-image would be wasteful.
kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)

# Accumulators.
all_stamps: list[np.ndarray] = []
all_labels: list[int] = []
all_det_x: list[float] = []
all_det_y: list[float] = []
all_mag: list[float] = []
all_visit: list[int] = []
all_sca: list[int] = []

per_file_counts = []

# ---- Process each (image, catalog) pair -----------------------------------
for i, (img_path, cat_path, visit, sca) in enumerate(pairs, 1):
    t_start = time.time()
    print(f"\n[{i}/{len(pairs)}] visit={visit} sca={sca}", flush=True)

    with fits.open(img_path) as hdul:
        image = hdul[0].data.astype(np.float32)
    Ny, Nx = image.shape

    _, median, _ = sigma_clipped_stats(image, sigma=3.0)
    convolved = convolve(image - median, kernel)
    finder = SourceFinder(n_pixels=10, progress_bar=False)
    segment_map = finder(convolved, threshold=DETECTION_THRESHOLD)
    if segment_map is None or segment_map.n_labels == 0:
        print("  no detections, skipping")
        continue
    catalog_obj = SourceCatalog(image - median, segment_map)
    det = catalog_obj.to_table(columns=["x_centroid", "y_centroid"])

    truth = Table.read(cat_path, format="ascii.commented_header")
    on_sca_mask = (truth["x"] >= 0) & (truth["x"] < Nx) & (truth["y"] >= 0) & (truth["y"] < Ny)
    truth_on = truth[on_sca_mask]
    truth_xy = np.column_stack([np.asarray(truth_on["x"]), np.asarray(truth_on["y"])])
    if len(truth_xy) == 0:
        print("  no on-SCA truth objects, skipping")
        continue

    tree = cKDTree(truth_xy)
    det_xy = np.column_stack([np.asarray(det["x_centroid"]), np.asarray(det["y_centroid"])])
    dist, idx = tree.query(det_xy, k=1)
    matched = dist < MATCH_TOL_PX

    n_kept = 0
    n_skipped_edge = 0
    n_skipped_class = 0
    for j in range(len(det)):
        if not matched[j]:
            continue
        truth_row = truth_on[int(idx[j])]
        obj_type = str(truth_row["obj_type"])
        if obj_type not in CLASS_TO_INT:
            n_skipped_class += 1
            continue

        x = float(det["x_centroid"][j])
        y = float(det["y_centroid"][j])
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = xi - HALF, xi + HALF
        y0, y1 = yi - HALF, yi + HALF
        if x0 < 0 or y0 < 0 or x1 > Nx or y1 > Ny:
            n_skipped_edge += 1
            continue

        all_stamps.append(image[y0:y1, x0:x1].astype(np.float32))
        all_labels.append(CLASS_TO_INT[obj_type])
        all_det_x.append(x)
        all_det_y.append(y)
        all_mag.append(float(truth_row["mag"]))
        all_visit.append(visit)
        all_sca.append(sca)
        n_kept += 1

    elapsed = time.time() - t_start
    per_file_counts.append((visit, sca, n_kept, n_skipped_edge, n_skipped_class))
    print(
        f"  kept {n_kept}  (skipped edge={n_skipped_edge}, class={n_skipped_class})"
        f"  [{elapsed:.1f}s]"
    )

# ---- Concatenate + save ---------------------------------------------------
if not all_stamps:
    raise SystemExit("No stamps extracted from any file.")

stamps_arr = np.stack(all_stamps, axis=0)
labels_arr = np.asarray(all_labels, dtype=np.int64)
det_x_arr = np.asarray(all_det_x, dtype=np.float32)
det_y_arr = np.asarray(all_det_y, dtype=np.float32)
mag_arr = np.asarray(all_mag, dtype=np.float32)
visit_arr = np.asarray(all_visit, dtype=np.int32)
sca_arr = np.asarray(all_sca, dtype=np.int32)

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"Total stamps: {len(stamps_arr)}")
print("\nClass distribution:")
for name, code in CLASS_TO_INT.items():
    n = int((labels_arr == code).sum())
    print(f"  {name:<10} ({code})  {n}")

print("\nPer-file breakdown:")
print(f"  {'visit':>6}  {'sca':>3}  {'kept':>5}  {'edge_skipped':>12}  {'class_skipped':>13}")
for visit, sca, n_kept, n_edge, n_class in per_file_counts:
    print(f"  {visit:>6}  {sca:>3}  {n_kept:>5}  {n_edge:>12}  {n_class:>13}")

np.savez_compressed(
    OUT_PATH,
    stamps=stamps_arr,
    labels=labels_arr,
    det_x=det_x_arr,
    det_y=det_y_arr,
    mag=mag_arr,
    visit=visit_arr,
    sca=sca_arr,
    class_names=np.array(list(CLASS_TO_INT.keys())),
)
print(f"\nSaved: {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
