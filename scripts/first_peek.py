"""First peek at a Roman OpenUniverse truth IMAGE.

Correction from round 1: this file is a 2D image, not a catalog table. Roman
SCAs are 4088x4088 pixels, and that's what's inside. In OpenUniverse,
"truth" images are the noiseless underlying scene the simulation is trying
to reproduce — the ideal sky BEFORE instrumental effects (read noise, cosmic
rays, PSF blur) get layered on. Perfect for a first visual look.

Filename decode: Roman_TDS_truth_F184_10307_17.fits.gz
  Roman_TDS  Roman Time-Domain Survey
  truth      noiseless truth image
  F184       Roman filter band F184 (near-IR, ~1.84 microns)
  10307      simulated visit / pointing ID
  17         SCA number (Roman has 18 detectors; this is chip #17)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

FITS_PATH = Path("data/openuniverse_preview/Roman_TDS_truth_F184_10307_17.fits.gz")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ---- Open the file and inspect ---------------------------------------------
with fits.open(FITS_PATH) as hdul:
    print("=" * 60)
    print("FITS structure:")
    print("=" * 60)
    hdul.info()

    header = hdul[0].header
    image = hdul[0].data  # shape (4088, 4088), float64

    # Print a curated set of header keys so we don't drown in all 66 of them.
    # If a key isn't present in the file, we just skip it. This is normal —
    # simulation FITS files often only carry a subset of the standard cards.
    interesting_keys = [
        "NAXIS1",
        "NAXIS2",
        "BUNIT",
        "FILTER",
        "EXPTIME",
        "DETECTOR",
        "INSTRUME",
        "TELESCOP",
        "RA_TARG",
        "DEC_TARG",
        "CRVAL1",
        "CRVAL2",
        "CD1_1",
        "CD2_2",
        "MJD-OBS",
        "OBSDATE",
    ]
    print("\nSelected header keywords (missing ones just aren't present):")
    for key in interesting_keys:
        if key in header:
            print(f"  {key:<10} {header[key]}")

# ---- Basic image stats -----------------------------------------------------
# We ignore non-finite pixels (NaN, inf) in the stats because a few of those
# can dominate min/max and give misleading numbers.
finite = image[np.isfinite(image)]
print("\nImage stats (finite pixels only):")
print(f"  shape        : {image.shape}")
print(f"  dtype        : {image.dtype}")
print(f"  min          : {finite.min():.4g}")
print(f"  max          : {finite.max():.4g}")
print(f"  mean         : {finite.mean():.4g}")
print(f"  median       : {np.median(finite):.4g}")
print(f"  99.9th %ile  : {np.percentile(finite, 99.9):.4g}")

# ---- Set up the display stretch -------------------------------------------
# Astronomical images have HUGE dynamic range: a bright star can be a million
# times brighter than the dim background. Linear display shows only the star.
# We fix this two ways:
#   - ZScaleInterval picks robust vmin/vmax based on pixel statistics
#     (typical of DS9 and other astro viewers).
#   - AsinhStretch is linear near zero and log-like for large values, so
#     both faint sources and bright ones show up in the same panel.
# ImageNormalize combines them into a matplotlib-compatible normalizer.
norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=AsinhStretch())

# ---- Plot 1: full frame ----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(image, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
ax.set_title(f"Roman SCA 17 truth image, F184  ({image.shape[0]}x{image.shape[1]})")
ax.set_xlabel("pixel x")
ax.set_ylabel("pixel y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "full_frame.png", dpi=120)
print(f"\nSaved: {FIGURES_DIR / 'full_frame.png'}")

# ---- Plot 2: zoom around the brightest object ------------------------------
# Cut a 512x512 window centered on the brightest pixel so you can actually
# see the object structure rather than a downsampled full frame.
brightest_y, brightest_x = np.unravel_index(np.argmax(image), image.shape)
print(f"\nBrightest pixel: (x={brightest_x}, y={brightest_y})  value={image.max():.4g}")

half = 256
y0, y1 = max(0, brightest_y - half), min(image.shape[0], brightest_y + half)
x0, x1 = max(0, brightest_x - half), min(image.shape[1], brightest_x + half)
zoom = image[y0:y1, x0:x1]

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
ax.set_title(f"Zoom around brightest pixel ({zoom.shape[0]}x{zoom.shape[1]})")
ax.set_xlabel("pixel x")
ax.set_ylabel("pixel y")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "zoom_brightest.png", dpi=120)
print(f"Saved: {FIGURES_DIR / 'zoom_brightest.png'}")

print("\nDone. Open the two PNGs.")
