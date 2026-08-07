"""Download OpenUniverse Roman TDS truth images + INDEX catalogs.

Fetches (image, catalog) pairs for a list of (visit, SCA) tuples from IRSA.
Skips files that are already on disk. Prints progress per file so we can see
what's happening on a big download.

File sources:
  Images:   .../RomanTDS/images/truth/F184/{visit}/Roman_TDS_truth_F184_{visit}_{sca}.fits.gz
  Catalogs: .../RomanTDS/truth/F184/{visit}/Roman_TDS_index_F184_{visit}_{sca}.txt

Note on data volume: each image is ~35 MB, each catalog ~3 MB.
Ten SCAs = ~400 MB.
"""

import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/roman/preview/RomanTDS"
BAND = "F184"
OUT_DIR = Path("data/openuniverse_preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (visit, sca) pairs to fetch. Spread across the visit-ID range to sample
# different-ish pointings. First entry is what we already have — the script
# will skip it if present on disk.
TARGETS = [
    (10307, 17),  # already have (baseline)
    (10307, 18),
    (10692, 17),
    (10692, 18),
    (11077, 17),
    (11077, 18),
    (12237, 17),
    (12237, 18),
    (13772, 17),
    (13772, 18),
]


def image_url(visit: int, sca: int) -> str:
    return f"{BASE}/images/truth/{BAND}/{visit}/Roman_TDS_truth_{BAND}_{visit}_{sca}.fits.gz"


def catalog_url(visit: int, sca: int) -> str:
    return f"{BASE}/truth/{BAND}/{visit}/Roman_TDS_index_{BAND}_{visit}_{sca}.txt"


def download_one(url: str, out_path: Path) -> tuple[bool, float]:
    """Download url -> out_path if not already present. Returns (downloaded, mb)."""
    if out_path.exists():
        mb = out_path.stat().st_size / 1e6
        return False, mb

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    t_start = time.time()
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    tmp_path.rename(out_path)
    mb = out_path.stat().st_size / 1e6
    elapsed = time.time() - t_start
    print(f"     downloaded {mb:.1f} MB in {elapsed:.1f}s ({mb/elapsed:.1f} MB/s)")
    return True, mb


total_downloaded_mb = 0.0
n_downloaded = 0
n_skipped = 0

for i, (visit, sca) in enumerate(TARGETS, 1):
    print(f"\n[{i}/{len(TARGETS)}] visit={visit} sca={sca}")

    for kind, url_fn, _suffix in [
        ("image", image_url, ".fits.gz"),
        ("catalog", catalog_url, ".txt"),
    ]:
        url = url_fn(visit, sca)
        filename = url.split("/")[-1]
        out_path = OUT_DIR / filename
        print(f"  {kind:<8}  {filename}", flush=True)
        try:
            was_new, mb = download_one(url, out_path)
        except RuntimeError as e:
            print(f"     ERROR: {e}", file=sys.stderr)
            continue
        if was_new:
            n_downloaded += 1
            total_downloaded_mb += mb
        else:
            n_skipped += 1
            print(f"     already present ({mb:.1f} MB), skipped")

print()
print("=" * 60)
print(f"Downloaded {n_downloaded} new files ({total_downloaded_mb:.1f} MB)")
print(f"Skipped    {n_skipped} already-present files")
print("=" * 60)
