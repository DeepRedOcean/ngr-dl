"""Detect sources in the Roman truth image with photutils.

Uses DAOStarFinder — a classic algorithm that finds point-source-like local
maxima above a threshold. Output is a table with (x_centroid, y_centroid,
flux, ...) for every detected source.

TUNING NOTE: this is a TRUTH image (noiseless). The textbook "5-sigma above
background" threshold from sigma-clipped stats is meaningless here because
the background std is basically zero — every faint PSF wing pixel registers
as a source. So we use an ABSOLUTE flux threshold (~100 counts) instead of
an N-sigma one, and a wider FWHM so each bright star is caught as ONE peak
instead of many. On a real noisy image, 5-sigma of the actual noise std is
the right choice and we'll use it there.

Docs: https://photutils.readthedocs.io/en/stable/user_guide/detection.html
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import photutils
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from photutils.detection import DAOStarFinder

# Opt into the new column naming (x_centroid, not xcentroid) — silences
# deprecation warnings and makes row-level access work correctly.
photutils.future_column_names = True

FITS_PATH = Path("data/openuniverse_preview/Roman_TDS_truth_F184_10307_17.fits.gz")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ---- Load the image --------------------------------------------------------
# We cast to float32 because DAOStarFinder does not need the extra precision
# and float32 uses half the memory (matters more later on real work).
with fits.open(FITS_PATH) as hdul:
    image = hdul[0].data.astype(np.float32)

# ---- Robust background statistics -----------------------------------------
# sigma_clipped_stats iteratively removes pixels beyond N-sigma (i.e. the
# bright sources) and computes the mean/median/std of the remaining
# "background" pixels. Standard first move before any source detection.
mean, median, std = sigma_clipped_stats(image, sigma=3.0)
print(f"Sigma-clipped background: mean={mean:.4g}  median={median:.4g}  std={std:.4g}")

# ---- Source detection ------------------------------------------------------
# FWHM = expected full-width-half-max of a point source in pixels. Roman WFI
# native scale is ~1.3 px FWHM, but truth images are often oversampled and
# stars have wide PSF wings. FWHM=5 catches each star as one wide peak
# rather than fragmenting into several detections.
#
# Threshold = absolute flux floor. On a real noisy image this would be
# ~5*std of the noise. Here the background is zero, so we use an absolute
# value based on the image's dynamic range: anything above 100 counts is
# unambiguously an object. This still gives us thousands of real sources
# without the false positives from PSF wings.
fwhm = 5.0
threshold = 100.0

finder = DAOStarFinder(fwhm=fwhm, threshold=threshold)
sources = finder(image - median)  # subtract background before searching

if sources is None or len(sources) == 0:
    print("No sources found. Try lowering threshold or adjusting FWHM.")
    raise SystemExit

print(f"Detected {len(sources)} sources  (threshold={threshold}, FWHM={fwhm}px)")

# Show top 10 brightest for a sanity peek.
sources.sort("flux", reverse=True)
print("\nTop 10 brightest sources:")
print(sources[:10]["id", "x_centroid", "y_centroid", "flux", "peak"])

# ---- Overlay detections on the full frame ---------------------------------
norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=AsinhStretch())

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(image, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
ax.scatter(
    sources["x_centroid"],
    sources["y_centroid"],
    s=8,
    facecolors="none",
    edgecolors="red",
    linewidths=0.3,
)
ax.set_title(f"Detected sources: {len(sources)}")
ax.set_xlabel("pixel x")
ax.set_ylabel("pixel y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "detections_full.png", dpi=140)
print(f"\nSaved: {FIGURES_DIR / 'detections_full.png'}")

# ---- Zoom around the brightest source --------------------------------------
brightest = sources[0]
bx, by = float(brightest["x_centroid"]), float(brightest["y_centroid"])

half = 256
y0, y1 = max(0, int(by - half)), min(image.shape[0], int(by + half))
x0, x1 = max(0, int(bx - half)), min(image.shape[1], int(bx + half))
zoom = image[y0:y1, x0:x1]

# Which detections fall inside this cutout?
mask = (
    (sources["x_centroid"] >= x0)
    & (sources["x_centroid"] < x1)
    & (sources["y_centroid"] >= y0)
    & (sources["y_centroid"] < y1)
)
local = sources[mask]
print(f"\nDetections inside zoom window: {len(local)}")

zoom_norm = ImageNormalize(zoom, interval=ZScaleInterval(), stretch=AsinhStretch())

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(
    zoom,
    origin="lower",
    cmap="gray",
    norm=zoom_norm,
    interpolation="nearest",
    extent=[x0, x1, y0, y1],
)
ax.scatter(
    local["x_centroid"],
    local["y_centroid"],
    s=80,
    facecolors="none",
    edgecolors="red",
    linewidths=1.0,
)
ax.set_title(f"Zoom around brightest source ({len(local)} detections in view)")
ax.set_xlabel("pixel x")
ax.set_ylabel("pixel y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "detections_zoom.png", dpi=140)
print(f"Saved: {FIGURES_DIR / 'detections_zoom.png'}")

print("\nDone. Open the two PNGs to see the detected sources highlighted.")
