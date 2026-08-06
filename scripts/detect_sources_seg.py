"""Segmentation-based source detection with photutils.

Unlike DAOStarFinder (point-source only), segmentation-based detection finds
contiguous groups of pixels above a threshold — a "segment" per object. It
catches both point sources (stars) AND extended sources (galaxies) with the
same algorithm, and produces a per-pixel object label map ("segmentation
map") that's directly useful downstream (postage stamp extraction, per-source
photometry, masking).

Pipeline:
  1. Detect connected pixels above a threshold  ->  SourceFinder
  2. Deblend overlapping detections            ->  built into SourceFinder
  3. Measure per-source properties             ->  SourceCatalog

Docs: https://photutils.readthedocs.io/en/stable/user_guide/segmentation.html
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import photutils
from astropy.convolution import convolve
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from photutils.segmentation import SourceCatalog, SourceFinder, make_2dgaussian_kernel

# Same future-column naming as the DAOStarFinder script.
photutils.future_column_names = True

FITS_PATH = Path("data/openuniverse_preview/Roman_TDS_truth_F184_10307_17.fits.gz")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ---- Load -----------------------------------------------------------------
with fits.open(FITS_PATH) as hdul:
    image = hdul[0].data.astype(np.float32)

mean, median, std = sigma_clipped_stats(image, sigma=3.0)
print(f"Sigma-clipped background: mean={mean:.4g}  median={median:.4g}  std={std:.4g}")

# ---- Matched-filter smoothing kernel --------------------------------------
# Convolving with a kernel matched to the expected source FWHM boosts real
# sources vs single-pixel noise. It's a mini matched-filter step before
# thresholding. Kernel FWHM roughly = expected source FWHM.
# In photutils >=3.0 the convolution is done EXPLICITLY before detection
# (SourceFinder no longer takes a kernel argument).
kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
data_bg_subtracted = image - median
convolved = convolve(data_bg_subtracted, kernel)

# ---- Detect + deblend -----------------------------------------------------
# SourceFinder is the one-shot wrapper: it thresholds the (convolved) image,
# connects pixels into segments, deblends overlapping detections, and
# returns a SegmentationImage (background=0, each object=unique integer label).
#
# n_pixels: minimum contiguous pixels above threshold for a valid source.
#   Filters out cosmic rays / hot pixels (usually 1-2 px). n_pixels=10 is
#   conservative — catches real sources, rejects noise blobs.
#
# threshold: absolute (same reasoning as detect_sources.py). On the truth
# image the background is zero; anything above 100 counts is unambiguous.
threshold = 100.0
finder = SourceFinder(n_pixels=10, progress_bar=False)
segment_map = finder(convolved, threshold)

if segment_map is None:
    print("No sources found. Try lowering threshold or npixels.")
    raise SystemExit

print(f"Segments (objects) detected: {segment_map.n_labels}")

# ---- Per-source measurements ----------------------------------------------
# SourceCatalog computes properties for each labeled segment: centroids,
# flux, morphology, ellipticity, etc.
catalog = SourceCatalog(data_bg_subtracted, segment_map)
sources = catalog.to_table(
    columns=[
        "label",
        "x_centroid",
        "y_centroid",
        "area",
        "segment_flux",
        "kron_flux",
        "elongation",
        "semimajor_axis",
    ]
)
sources.sort("segment_flux", reverse=True)

print("\nTop 10 brightest sources:")
print(sources[:10])

# ---- Full-frame visualization: image + segmentation map side by side ------
norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=AsinhStretch())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.imshow(image, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
ax1.scatter(
    sources["x_centroid"],
    sources["y_centroid"],
    s=10,
    facecolors="none",
    edgecolors="red",
    linewidths=0.4,
)
ax1.set_title(f"Image + detections ({segment_map.n_labels} sources)")
ax1.set_xlabel("pixel x")
ax1.set_ylabel("pixel y")

ax2.imshow(segment_map.data, origin="lower", cmap=segment_map.cmap, interpolation="nearest")
ax2.set_title("Segmentation map (each color = one object)")
ax2.set_xlabel("pixel x")
ax2.set_ylabel("pixel y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "segmentation_full.png", dpi=140)
print(f"\nSaved: {FIGURES_DIR / 'segmentation_full.png'}")

# ---- Zoom around the brightest source (same window as DAOStarFinder) -----
brightest = sources[0]
bx, by = float(brightest["x_centroid"]), float(brightest["y_centroid"])

half = 256
y0, y1 = max(0, int(by - half)), min(image.shape[0], int(by + half))
x0, x1 = max(0, int(bx - half)), min(image.shape[1], int(bx + half))
zoom = image[y0:y1, x0:x1]
zoom_seg = segment_map.data[y0:y1, x0:x1]

mask = (
    (sources["x_centroid"] >= x0)
    & (sources["x_centroid"] < x1)
    & (sources["y_centroid"] >= y0)
    & (sources["y_centroid"] < y1)
)
local = sources[mask]
print(f"\nDetections inside zoom window: {len(local)}")

zoom_norm = ImageNormalize(zoom, interval=ZScaleInterval(), stretch=AsinhStretch())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.imshow(
    zoom,
    origin="lower",
    cmap="gray",
    norm=zoom_norm,
    interpolation="nearest",
    extent=[x0, x1, y0, y1],
)
ax1.scatter(
    local["x_centroid"],
    local["y_centroid"],
    s=80,
    facecolors="none",
    edgecolors="red",
    linewidths=1.0,
)
ax1.set_title(f"Zoom + detections ({len(local)} in view)")
ax1.set_xlabel("pixel x")
ax1.set_ylabel("pixel y")

ax2.imshow(
    zoom_seg,
    origin="lower",
    cmap=segment_map.cmap,
    interpolation="nearest",
    extent=[x0, x1, y0, y1],
)
ax2.set_title("Segmentation labels (zoom)")
ax2.set_xlabel("pixel x")
ax2.set_ylabel("pixel y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "segmentation_zoom.png", dpi=140)
print(f"Saved: {FIGURES_DIR / 'segmentation_zoom.png'}")

print("\nDone. Open the two PNGs to compare vs. DAOStarFinder.")
