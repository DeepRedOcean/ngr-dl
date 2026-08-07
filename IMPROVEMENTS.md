# Improvements & Deferred Work

A living document of ideas, upgrades, and "we noticed X but didn't fix it"
items that surfaced during development. Claude adds items here as they come
up (proactively — no need to be asked). Human reviews when planning what to
build next.

Ordering within each section: rough priority, top = most useful next.

---

## Data & I/O

- **Grab a matching noisy image** (`Roman_TDS_simple_F184_10307_17.fits.gz`
  or similar naming) to compare with the truth image. Lets us see actual
  Roman noise texture and validates that our detection pipeline behaves
  differently on real vs. noiseless data.
- **Full WCS handling / sky-coord cross-match**. Current cross-match script
  works in *pixel space* because OpenUniverse truth catalogs carry pixel
  coords for the target SCA. That won't hold for non-OpenUniverse catalogs
  or for multi-epoch matching (different SCAs, different WCS). Need proper
  `astropy.wcs.WCS` + `SkyCoord.match_to_catalog_sky` when we get there —
  probably at Stage 0 SSO filter time.
- **Truth catalog download + cross-match**. There's a matching truth catalog
  (a Parquet or FITS table of every object placed in this pointing) somewhere
  in OpenUniverse. Cross-matching against it gives us completeness/purity
  metrics for whatever detector we're using.
- **Refactor data I/O into `src/ngr_dl/data.py`**. Once we have more than 2
  files to manage, the ad-hoc `Path(...)` at the top of scripts becomes a
  liability. Central loading + validation module.
- **Download automation**. Right now data arrived by hand in Downloads. Once
  we know which files we actually want, script this via `wget` / `requests`
  / IRSA API so the pipeline is reproducible from scratch.

## Source detection

- **Migrate `photutils` → `sep`** (SExtractor Python port) once we outgrow
  `photutils.segmentation`. `sep` is faster, has better adaptive background
  estimation, and is the historical gold standard used by SDSS/DES/LSST/HST
  pipelines. Small extra install.
- **Metadata-derived detection thresholds**. Read `RDNOISE`, gain, exposure
  time, sky background from FITS headers and predict expected per-pixel noise
  instead of relying on sigma-clipped stats. What production observatory
  pipelines do. Requires OpenUniverse noisy images (which carry proper
  metadata) rather than truth images (which don't).
- **ML-based source detection** (DeepSource, etc.) as a stretch goal. Probably
  overkill for standard sources, but potentially useful for detecting unusual
  morphologies (SSO streaks, extended low-surface-brightness features) that
  classical thresholding misses.
- **GPU-accelerated detection via cuCIM** if we ever process large numbers of
  images. `cucim.skimage.measure.label` + GPU watershed would give ~3-5x on
  segmentation. Not worth it now (photutils is fast enough for one-off use)
  but a good escape hatch if we need to detect across hundreds of pointings.

## Modeling

- **Data augmentation for the typer** — random rotations (any angle;
  orientation is meaningless for these objects), h/v flips, small pixel
  translations. Current typer overfits at ~90% train accuracy with weak
  ~60% star recall; augmentation should improve generalization on the
  small (~70 train stars) dataset.
- **Multi-band input for the typer** — currently trained on F184 only, so
  the model has no color information. Roman has 7 filters; feeding several
  bands as channels would resolve the "compact galaxy looks like a star"
  false positives that dominate our confident errors.
- **Per-visit train/test split for the typer** — current stratified split
  is class-only; when we have multi-visit data (already downloaded), a
  proper eval should hold out ENTIRE visits, not random samples, because
  same-visit stamps are spatially correlated. E.g. train on visits
  {10307, 10692, 11077, 12237} and test on {13772}. Would give a more
  honest generalization number.
- **More Roman images (COMPLETED 2026-08-07)** — downloaded 5 visits x
  2 SCAs. Kept in the queue as historical note.
- **BatchNorm alternatives on tiny datasets** — the training instability
  at epochs 14-16 (val_loss spiking while train_loss fell) is a known
  BN-with-small-batch pathology. GroupNorm or LayerNorm would be more
  stable at our current batch size (32) and small val set (306).
- **Wider stamps around confused sources** — some missed stars had nearby
  galaxies in the 64px window; a larger context might help the model
  distinguish "point source at center" from "extended source at center".
- **Segmentation-masked stamps for the typer** — cut the stamp as usual but
  zero out (or mask) pixels that don't belong to the source's own segment.
  Directly kills context pollution from nearby unrelated objects, which was
  the visible failure mode on 5+ of our 10 missed stars. Cheaper than a
  cascade specialist and same intent. Alternative: two-channel input (raw
  stamp + binary segment mask) so the model can learn to attend to the
  source pixels while still seeing local background.
- **Cascade specialist for star-near-galaxy classification** — a separate
  model for sources embedded in or immediately adjacent to detected galaxy
  segments. Worth revisiting when we have (a) multi-image data with more
  such cases, (b) real physical superpositions (foreground stars on
  background galaxies) as a well-populated training class. Current
  ~117-star dataset isn't enough to train yet another specialist without
  starving it of examples.

## Findings from Stage 1 typer v1 (2026-08-07)

- **Data scale was the primary constraint.** Going from 117 to 721 stars
  (5x visits) pushed star recall from 0.58 -> 0.96 and F1 from 0.68 -> 0.97.
  Confirmed empirically: the CNN architecture is capable, it just needed
  more training examples. Diminishing returns likely start soon on THIS
  model, so next data expansion probably has less leverage than augmentation
  or multi-band.
- **Remaining 4% failures are at the depth limit.** All 6 missed stars in
  v1 test set are mag 4.7-5.4 (faintest bin only). This is a Roman F184
  photon-noise floor issue, not a model issue. Would need brighter data or
  multi-band info to push further.
- **v1 model is well-calibrated in its errors.** False-positive galaxies
  in v0 were confident (P=0.94); in v1 they're borderline (P=0.64, 0.82).
  This means confidence thresholding could route ambiguous cases to human
  review — a feature for the eventual anomaly pipeline.
- **The 2 false-positive galaxies in v1 test set look visually unusual**
  (extended, distorted, or with diffuse halos). They may be interesting
  anomalies in their own right (interacting galaxies, low-surface-brightness
  features). Confidence-based routing to a human queue is more valuable than
  trying to force them into "galaxy" bin.

## Architecture (from threshold sweep 2026-07-31)

- **Per-specialist detection thresholds**. The threshold sweep showed that
  different object types have wildly different optimal detection thresholds:
  stars peak at threshold ~20, galaxies at ~1, transients at ~3. A single
  global threshold is a compromise for everything. When we build the Stage 2
  specialists, each should ideally run detection at its own optimal setting
  (or use a lower-threshold candidate list and let the classifier reject
  false positives per type).
- **Simple segmentation ceiling at ~55% completeness**. Even at the best
  threshold, segmentation misses ~45% of truth objects — sub-noise faint
  sources, heavily blended pairs, giant merged blobs. This ceiling motivates
  the eventual move to ML-based detection (see "ML-based source detection"
  above). Not a today problem but a real long-term motivator.

## Infrastructure

- **CI slowness**: ~3–5 min due to ~2GB torch CUDA wheel downloads. Consider
  a CPU-only torch override in the CI environment (`--extra-index-url` with
  the CPU wheel) since CI only runs ruff + pytest smoke, no actual training.
- **Document Windows-specific hacks in README**: the pre-commit cache
  relocation via `PRE_COMMIT_HOME` was a Windows-specific workaround for
  Defender file locks. Future contributors will hit the same wall.
- **pytest coverage reporting** in CI once we have real test content beyond
  the smoke test.
- **Optional: pre-commit autoupdate as a scheduled workflow** so hook versions
  don't drift over months of inactivity.

## Anomaly detection (project's main story)

- *(empty for now — will fill as we design the transient specialist)*

## Known bugs / minor issues

- `xcentroid` vs `x_centroid` column naming in photutils changed between
  versions. Currently opting into the new naming via
  `photutils.future_column_names = True`. Remove once photutils >=4.0 is
  standard (old names will be removed then anyway).
