"""
prepare_combined_dataset.py

Fuse Sentinel-1 (Lee-filtered) and Sentinel-2 (RGB + MNDWI) with cloud mask
into 7-band GeoTIFFs for the s1s2-water dataset.

Output band order (all float32):
  1) Red        – S2 reflectance (DN/10000), clipped to [0, 1]
  2) Green      – S2 reflectance (DN/10000), clipped to [0, 1]
  3) Blue       – S2 reflectance (DN/10000), clipped to [0, 1]
  4) MNDWI      – (Green-SWIR1)/(Green+SWIR1+eps), clipped to [0, 1]
  5) VV_db      – S1 Lee-filtered dB, clipped to [-25, 5], min-max scaled to [0, 1]
  6) VH_db      – S1 Lee-filtered dB, clipped to [-32, -5], min-max scaled to [0, 1]
  7) cloud_mask – inverted (cloud=1, no-cloud=0), [0, 1]

Memory-optimised: bands are read, processed, and written one at a time.
Peak RAM ≈ one Lee filter pass (~8 GB for 10980×10980) instead of ~16 GB.

Usage:
  python prepare_combined_dataset.py --data_root "D:\\s1s2_water\\data"
  python prepare_combined_dataset.py --data_root "D:\\s1s2_water\\data" --only 1,71 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from prepare.sar_lee_filter_dbx import (
    EPS,
    _valid_mask,
    db_to_linear_safe,
    lee_filter_basic,
    lee_filter_refined,
    linear_to_db_safe,
)

BANDS = ["red", "green", "blue", "mndwi", "vv", "vh", "cloud_mask"]

# Fixed dB clipping bounds for SAR bands
VV_BOUNDS = (-25.0, 5.0)
VH_BOUNDS = (-32.0, -5.0)
DB_CLIP = {5: VV_BOUNDS, 6: VH_BOUNDS}  # output band index → (lo, hi)

# ---------------------------------------------------------------------------
# Resampling (single-band)
# ---------------------------------------------------------------------------

def _resample_band(
    band: np.ndarray,
    src_transform,
    src_crs,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    resampling: Resampling,
) -> np.ndarray:
    """Resample a single 2-D band to dst grid. Returns input unchanged if shapes match."""
    if band.shape == dst_shape:
        return band
    dst = np.empty(dst_shape, dtype=band.dtype)
    reproject(
        source=band,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst


# ---------------------------------------------------------------------------
# Strip-based Lee filter (reduces peak RAM from ~15 GB to ~500 MB)
# ---------------------------------------------------------------------------

STRIP_HEIGHT = 512  # rows per strip (tune for your machine)


def _apply_filter_striped(
    img_linear: np.ndarray,
    filter_fn,
    ksize: int,
    enl: float,
    strip_height: int = STRIP_HEIGHT,
) -> np.ndarray:
    """Apply a Lee filter in horizontal strips with overlap to reduce peak memory.

    Each strip includes ksize//2 extra rows on top and bottom so that the
    convolution boundary is always covered by real pixels. Only the central
    (non-padded) rows of each filtered strip are kept.
    """
    H, W = img_linear.shape
    pad = ksize // 2
    out = np.empty((H, W), dtype=np.float32)

    for y0 in range(0, H, strip_height):
        y1 = min(y0 + strip_height, H)

        # padded read bounds
        py0 = max(y0 - pad, 0)
        py1 = min(y1 + pad, H)

        strip = img_linear[py0:py1, :].copy()
        filtered = filter_fn(strip, ksize=ksize, enl=enl, nodata=np.nan)

        # trim the overlap padding
        trim_top = y0 - py0
        trim_bot = py1 - y1
        if trim_bot == 0:
            out[y0:y1, :] = filtered[trim_top:, :]
        else:
            out[y0:y1, :] = filtered[trim_top:-trim_bot, :]

        del strip, filtered
        gc.collect()

    return out


# ---------------------------------------------------------------------------
# S1: raw Int16 band → Lee-filtered dB (float32)
# ---------------------------------------------------------------------------

def _s1_filter_band(
    s1_path: Path,
    s1_band_1based: int,
    s1_nodata: float | None,
    s1_transform,
    s1_crs,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    db_scale: float,
    ksize: int,
    enl: float,
    filter_fn,
) -> np.ndarray:
    """Read one S1 band, resample, convert dB→linear, apply *filter_fn*, return dB float32."""
    with rasterio.open(s1_path) as src:
        raw = src.read(s1_band_1based, out_dtype="float32")

    raw = _resample_band(raw, s1_transform, s1_crs,
                         dst_shape, dst_transform, dst_crs,
                         Resampling.bilinear)

    valid = _valid_mask(raw, s1_nodata)
    db = np.empty(raw.shape, dtype=np.float32)
    db[:] = np.nan
    db[valid] = raw[valid] / db_scale
    del raw
    gc.collect()

    lin = db_to_linear_safe(db, valid)
    del db, valid
    gc.collect()

    lin_f = _apply_filter_striped(lin, filter_fn, ksize=ksize, enl=enl)
    del lin
    gc.collect()

    db_f = linear_to_db_safe(lin_f)
    del lin_f
    return db_f


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _make_output_profile(s2_profile: dict) -> dict:
    """Build the 7-band float32 output profile from the S2 source profile."""
    prof = dict(s2_profile)
    prof.update(
        count=7,
        dtype="float32",
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    prof.pop("nodata", None)
    return prof


def _finalise_tags(dst) -> None:
    """Set band descriptions and tags on an open rasterio dataset."""
    for i, name in enumerate(BANDS, 1):
        dst.set_band_description(i, name)
    dst.update_tags(**{f"band_{i}": name for i, name in enumerate(BANDS, 1)})


# ---------------------------------------------------------------------------
# Per-grid processing  (streaming, band-by-band)
# ---------------------------------------------------------------------------

def _process_grid(
    grid_dir: Path,
    grid_id: str,
    out_dir_basic: Path,
    out_dir_refined: Path,
    ksize: int,
    enl: float,
    db_scale: float,
    overwrite: bool,
) -> tuple[str | None, str | None]:
    """Process one grid. Returns (skip_reason, msk_relative_path).

    skip_reason is None on success, or a string explaining why it was skipped.
    msk_relative_path is e.g. '1/sentinel12_s2_1_msk.tif' on success, else None.
    """
    out_basic = out_dir_basic / f"sen12_{grid_id}.tif"
    out_refined = out_dir_refined / f"sen12_{grid_id}.tif"

    if not overwrite and out_basic.exists() and out_refined.exists():
        return "outputs already exist (use --overwrite)", None

    # --- locate input files ---
    s1_path = grid_dir / f"sentinel12_s1_{grid_id}_img.tif"
    s2_path = grid_dir / f"sentinel12_s2_{grid_id}_img.tif"
    msk_path = grid_dir / f"sentinel12_s2_{grid_id}_valid.tif"

    for p, label in [(s1_path, "S1 img"), (s2_path, "S2 img"), (msk_path, "S2 valid mask")]:
        if not p.exists():
            return f"missing {label}: {p.name}", None

    # --- collect metadata (no pixel data yet) ---
    with rasterio.open(s2_path) as src:
        s2_profile = dict(src.profile)
        s2_transform = src.transform
        s2_crs = src.crs
        s2_shape = (src.height, src.width)
        if src.count != 6:
            return f"S2 has {src.count} bands, expected 6", None

    with rasterio.open(s1_path) as src:
        s1_nodata = src.nodata
        s1_transform = src.transform
        s1_crs = src.crs
        if src.count != 2:
            return f"S1 has {src.count} bands, expected 2", None

    with rasterio.open(msk_path) as src:
        msk_transform = src.transform
        msk_crs = src.crs

    out_prof = _make_output_profile(s2_profile)
    out_dir_basic.mkdir(parents=True, exist_ok=True)
    out_dir_refined.mkdir(parents=True, exist_ok=True)

    # Open both outputs for the entire duration so we can write bands one by one.
    with rasterio.open(out_basic, "w", **out_prof) as dst_b, \
         rasterio.open(out_refined, "w", **out_prof) as dst_r:

        # ---- bands 1-3: R, G, B from S2 (read individually) ----
        # band mapping: S2 file 1-based → (output band, need to keep?)
        #   S2 band 3 (Red)   → output 1
        #   S2 band 2 (Green) → output 2  (keep for MNDWI)
        #   S2 band 1 (Blue)  → output 3
        # Sentinel-2 L1C TOA reflectance scaled by 10000 → divide by 10000
        scale = np.float32(10000.0)

        with rasterio.open(s2_path) as src:
            # Red – clip to [0, 1]
            red = src.read(3, out_dtype="float32")
            np.divide(red, scale, out=red)
            np.clip(red, 0.0, 1.0, out=red)
            dst_b.write(red, 1)
            dst_r.write(red, 1)
            del red

            # Green – clip to [0, 1], keep for MNDWI
            green = src.read(2, out_dtype="float32")
            np.divide(green, scale, out=green)
            np.clip(green, 0.0, 1.0, out=green)
            dst_b.write(green, 2)
            dst_r.write(green, 2)

            # Blue – clip to [0, 1]
            blue = src.read(1, out_dtype="float32")
            np.divide(blue, scale, out=blue)
            np.clip(blue, 0.0, 1.0, out=blue)
            dst_b.write(blue, 3)
            dst_r.write(blue, 3)
            del blue

            # ---- band 4: MNDWI ----
            swir1 = src.read(5, out_dtype="float32")
            np.divide(swir1, scale, out=swir1)

        # mndwi = (green - swir1) / (green + swir1 + eps)  — in-place as much as possible
        denom = green + swir1
        denom += EPS
        mndwi = green - swir1
        del green, swir1
        mndwi /= denom
        del denom

        # clip MNDWI to [0, 1]
        np.clip(mndwi, 0.0, 1.0, out=mndwi)
        dst_b.write(mndwi, 4)
        dst_r.write(mndwi, 4)
        del mndwi
        gc.collect()

        # ---- bands 5-6: VV / VH  (one filter at a time, re-read from disk) ----
        for s1_idx, out_idx in [(1, 5), (2, 6)]:  # 1-based: VV=1, VH=2
            lo, hi = DB_CLIP[out_idx]

            # basic Lee → clip dB → min-max scale to [0, 1] → write to dst_b
            db_f = _s1_filter_band(
                s1_path, s1_idx, s1_nodata,
                s1_transform, s1_crs,
                s2_shape, s2_transform, s2_crs,
                db_scale, ksize, enl,
                lee_filter_basic,
            )
            np.clip(db_f, lo, hi, out=db_f)
            db_f = (db_f - lo) / (hi - lo)
            dst_b.write(db_f, out_idx)
            del db_f
            gc.collect()

            # refined Lee → clip dB → min-max scale to [0, 1] → write to dst_r
            db_f = _s1_filter_band(
                s1_path, s1_idx, s1_nodata,
                s1_transform, s1_crs,
                s2_shape, s2_transform, s2_crs,
                db_scale, ksize, enl,
                lee_filter_refined,
            )
            np.clip(db_f, lo, hi, out=db_f)
            db_f = (db_f - lo) / (hi - lo)
            dst_r.write(db_f, out_idx)
            del db_f
            gc.collect()

        # ---- band 7: cloud mask (invert: src has cloud=0 → output cloud=1) ----
        with rasterio.open(msk_path) as src:
            msk = src.read(1, out_dtype="float32")
        msk = _resample_band(msk, msk_transform, msk_crs,
                             s2_shape, s2_transform, s2_crs,
                             Resampling.nearest)
        np.subtract(1.0, msk, out=msk)
        dst_b.write(msk, 7)
        dst_r.write(msk, 7)
        del msk

        # ---- finalise metadata ----
        _finalise_tags(dst_b)
        _finalise_tags(dst_r)

    msk_rel = f"{grid_id}/{msk_path.name}"
    return None, msk_rel


# ---------------------------------------------------------------------------
# Grid discovery
# ---------------------------------------------------------------------------

def _discover_grids(data_root: Path) -> list[str]:
    """Find numeric subdirectories under data_root."""
    grids = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.isdigit():
            grids.append(p.name)
    return grids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fuse S1 (Lee-filtered) + S2 (RGB, MNDWI) + cloud mask into 7-band GeoTIFFs.",
    )
    ap.add_argument("--data_root", type=str, default="./data",
                     help="Root data directory containing grid folders (default: ./data)")
    ap.add_argument("--ksize", type=int, default=7,
                     help="Lee filter kernel size (default: 7)")
    ap.add_argument("--enl", type=float, default=4.0,
                     help="Equivalent Number of Looks (default: 4.0)")
    ap.add_argument("--db_scale", type=float, default=100.0,
                     help="Scale factor of stored S1 dB values (default: 100)")
    ap.add_argument("--only", type=str, default=None,
                     help="Comma-separated grid IDs to process (default: all)")
    ap.add_argument("--overwrite", action="store_true",
                     help="Overwrite existing output files")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"ERROR: data_root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    out_dir_basic = data_root / "data_combined" / "img_basic_lee"
    out_dir_refined = data_root / "data_combined" / "img_refined_lee"

    # discover grids
    all_grids = _discover_grids(data_root)
    if not all_grids:
        print(f"ERROR: no numeric grid folders found under {data_root}", file=sys.stderr)
        sys.exit(1)

    # optional subset
    if args.only:
        subset = set(args.only.split(","))
        grids = [g for g in all_grids if g in subset]
        missing = subset - set(grids)
        if missing:
            print(f"WARNING: requested grids not found: {sorted(missing)}")
    else:
        grids = all_grids

    print(f"Found {len(grids)} grid(s) to process (ksize={args.ksize}, enl={args.enl}, db_scale={args.db_scale})")
    print(f"Output dirs:")
    print(f"  basic_lee:   {out_dir_basic}")
    print(f"  refined_lee: {out_dir_refined}")
    print()

    processed = 0
    skipped = 0
    lee_lines: list[str] = []
    ref_lines: list[str] = []
    t0 = time.time()

    for i, gid in enumerate(grids, 1):
        grid_dir = data_root / gid
        print(f"[{i}/{len(grids)}] Processing grid {gid} ... ", end="", flush=True)

        try:
            reason, msk_rel = _process_grid(
                grid_dir, gid,
                out_dir_basic, out_dir_refined,
                ksize=args.ksize,
                enl=args.enl,
                db_scale=args.db_scale,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            reason, msk_rel = f"error: {exc}", None

        if reason is None:
            print("OK")
            processed += 1
            lee_lines.append(f"data_combined/img_basic_lee/sen12_{gid}.tif {msk_rel}")
            ref_lines.append(f"data_combined/img_refined_lee/sen12_{gid}.tif {msk_rel}")
        else:
            print(f"SKIPPED ({reason})")
            skipped += 1

    # Append new entries to mapping files (preserves previous runs)
    out_combined = data_root / "data_combined"
    out_combined.mkdir(parents=True, exist_ok=True)
    for fname, new_lines in [
        ("img_msk_lee_mapping.txt", lee_lines),
        ("img_msk_refinedLee_mapping.txt", ref_lines),
    ]:
        if not new_lines:
            continue
        fpath = out_combined / fname
        with open(fpath, "a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")
        print(f"  {fname}: appended {len(new_lines)} entries (total {len(fpath.read_text(encoding='utf-8').splitlines())})")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  |  processed: {processed}  |  skipped: {skipped}")


if __name__ == "__main__":
    main()
