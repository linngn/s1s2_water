import json
import logging
import numpy as np
import sys
import tifffile as tiff
import tqdm

import rasterio
from rasterio.windows import Window
from pathlib import Path
from prepare.utils import scale_min_max, tile_array
from pystac_client import Client
from ukis_pysat.raster import Image


def run(data_dir, out_dir, sensor="s1", tile_shape=(256, 256), img_bands_idx=[0, 1], slope=False, exclude_nodata=False):
    logging.info("Splitting training samples")

    if Path(Path(data_dir) / "catalog.json").is_file:
        catalog = Client.open(Path(data_dir) / "catalog.json")
    else:
        raise NotImplementedError("Cannot find catalog.json file in data_dir")

    if sensor == "s1":
        scale_min, scale_max = 0, 100.0
    elif sensor == "s2":
        scale_min, scale_max = 0, 10000.0
    else:
        raise NotImplementedError(f"Sensor {str(sensor)} not supported ['s1', 's2']")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "train/img").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "train/msk").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "test/img").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "test/msk").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "val/img").mkdir(parents=True, exist_ok=True)
    Path(Path(out_dir) / "val/msk").mkdir(parents=True, exist_ok=True)

    items = [item.to_dict() for item in catalog.get_all_items()]
    sys.stdout.flush()
    for i, item in tqdm.tqdm(enumerate(items), total=len(items)):
        split = item["properties"]["split"]
        subdir = Path(item["assets"][f"{sensor}_img"]["href"]).parent.name
        msk_file = Path(data_dir) / Path(subdir) / Path(item["assets"][f"{sensor}_msk"]["href"]).name
        valid_file = Path(data_dir) / Path(subdir) / Path(item["assets"][f"{sensor}_valid"]["href"]).name
        slope_file = Path(data_dir) / Path(subdir) / Path(item["assets"]["copdem30_slope"]["href"]).name
        img_file = Path(data_dir) / Path(subdir) / Path(item["assets"][f"{sensor}_img"]["href"]).name

        msk = Image(data=msk_file, dimorder="last")
        valid = Image(data=valid_file, dimorder="last")
        slope = Image(data=slope_file, dimorder="last") if slope else None
        img = Image(data=img_file, dimorder="last")
        img_scaled = scale_min_max(img.arr[:, :, img_bands_idx], min=scale_min, max=scale_max)

        if slope:
            slope.warp(resampling_method=2, dst_crs=img.dataset.crs, target_align=img)
            img_scaled = np.append(img_scaled, slope.arr, axis=2)

        img_tiles = tile_array(img_scaled, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)
        msk_tiles = tile_array(msk.arr, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)
        valid_tiles = (
            tile_array(valid.arr, xsize=tile_shape[0], ysize=tile_shape[1], overlap=0.0, padding=False)
            if exclude_nodata
            else None
        )

        for j in range(len(img_tiles)):
            if exclude_nodata:
                if 0 in valid_tiles[j, :, :, :]:
                    continue
            tiff.imsave(
                Path(out_dir) / f"{split}/img/{Path(img_file).stem}_{j}.tif",
                img_tiles[j, :, :, :],
                planarconfig="contig",
            )
            tiff.imsave(Path(out_dir) / f"{split}/msk/{Path(msk_file).stem}_{j}.tif", msk_tiles[j, :, :, :])


# ---------------------------------------------------------------------------
# split_train_dataset  –  fused 7-band images + mask → patches + train/val
# ---------------------------------------------------------------------------

def _read_mapping(path):
    """Parse a mapping txt file. Returns list of (img_filename, msk_rel_path)."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            entries.append((parts[0], parts[1]))
    return entries


def split_train_dataset(
    data_dir,
    out_dir,
    patch_shape=(256, 256),
    val_ratio=0.2,
    seed=42,
):
    """Tile fused 7-band images and masks into patches, split train / val.

    Parameters
    ----------
    data_dir : str or Path
        ROOT/data directory containing grid folders and
        data_combined/ with fused images and mapping files.
    out_dir : str or Path
        Output directory for patches and split txt files.
    patch_shape : tuple of int
        (height, width) of each patch.
    val_ratio : float
        Fraction of *scenes* assigned to validation.
    seed : int
        Random seed for reproducible train/val split.

    Reads
    -----
    data_dir/data_combined/img_msk_lee_mapping.txt
    data_dir/data_combined/img_msk_refinedLee_mapping.txt

    Writes
    ------
    out_dir/image/              – basic-Lee image patches
    out_dir/image_refined/      – refined-Lee image patches
    out_dir/mask/               – mask patches (shared)
    out_dir/train_data.txt
    out_dir/validating_data.txt
    out_dir/train_data_refined.txt
    out_dir/validating_data_refined.txt
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    combined_dir = data_dir / "data_combined"
    ph, pw = patch_shape

    # --- read mapping files ---
    lee_map = _read_mapping(combined_dir / "img_msk_lee_mapping.txt")
    ref_map = _read_mapping(combined_dir / "img_msk_refinedLee_mapping.txt")
    if not lee_map:
        raise ValueError("img_msk_lee_mapping.txt is empty or missing")
    if not ref_map:
        raise ValueError("img_msk_refinedLee_mapping.txt is empty or missing")

    # --- create output directories ---
    img_dir = out_dir / "image"
    img_ref_dir = out_dir / "image_refined"
    msk_dir = out_dir / "mask"
    for d in (img_dir, img_ref_dir, msk_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- tile each scene (windowed I/O – only one patch in RAM at a time) ---
    scene_patches = {}  # stem -> [patch_name, ...]

    for idx, (img_rel, msk_rel) in enumerate(lee_map):
        stem = Path(img_rel).stem  # e.g. "sen12_1"

        # mapping has full relative paths, e.g.
        #   data_combined/img_basic_lee/sen12_1.tif  1/sentinel12_s2_1_msk.tif
        basic_path = data_dir / img_rel
        # derive refined path by swapping the directory
        refined_path = data_dir / img_rel.replace("img_basic_lee", "img_refined_lee")
        msk_path = data_dir / msk_rel

        # verify files exist
        missing = [p for p in (basic_path, refined_path, msk_path) if not p.exists()]
        if missing:
            print(f"[{idx+1}/{len(lee_map)}] {stem} ... SKIP (missing {[p.name for p in missing]})")
            continue

        print(f"[{idx+1}/{len(lee_map)}] Tiling {stem} ... ", end="", flush=True)

        patch_names = []

        with rasterio.open(basic_path) as src_b, \
             rasterio.open(refined_path) as src_r, \
             rasterio.open(msk_path) as src_m:

            H, W = src_b.height, src_b.width

            # dimension check
            if (src_r.height, src_r.width) != (H, W):
                print("SKIP (refined shape mismatch)")
                continue
            if (src_m.height, src_m.width) != (H, W):
                print(f"SKIP (mask shape {src_m.height}x{src_m.width} != {H}x{W})")
                continue

            nrows = H // ph
            ncols = W // pw

            # build patch profiles from source metadata
            img_prof = dict(src_b.profile)
            img_prof.update(height=ph, width=pw)

            msk_prof = dict(src_m.profile)
            msk_prof.update(height=ph, width=pw)

            # band descriptions from source
            img_descriptions = [src_b.descriptions[i] for i in range(src_b.count)]

            patch_idx = 0
            for r in range(nrows):
                for c in range(ncols):
                    patch_idx += 1
                    win = Window(c * pw, r * ph, pw, ph)
                    win_transform = src_b.window_transform(win)

                    p_basic = src_b.read(window=win)       # (bands, ph, pw)
                    p_refined = src_r.read(window=win)      # (bands, ph, pw)
                    p_mask = src_m.read(window=win)          # (1, ph, pw)

                    patch_name = f"{stem}_{patch_idx}.tif"

                    # write image patches (inherit profile + band descriptions)
                    ip = {**img_prof, "transform": win_transform}
                    for out_path, data in [(img_dir / patch_name, p_basic),
                                           (img_ref_dir / patch_name, p_refined)]:
                        with rasterio.open(out_path, "w", **ip) as dst:
                            dst.write(data)
                            for bi, desc in enumerate(img_descriptions, 1):
                                if desc:
                                    dst.set_band_description(bi, desc)

                    # write mask patch (inherit profile)
                    mp = {**msk_prof, "transform": win_transform}
                    with rasterio.open(msk_dir / patch_name, "w", **mp) as dst:
                        dst.write(p_mask)

                    patch_names.append(patch_name)

        scene_patches[stem] = patch_names
        print(f"{len(patch_names)} patches")

    if not scene_patches:
        print("ERROR: no scenes were tiled")
        return

    # --- randomly split patches into train / val ---
    all_patches = []
    for stem in sorted(scene_patches.keys()):
        for pname in scene_patches[stem]:
            all_patches.append(pname)

    rng = np.random.RandomState(seed)
    rng.shuffle(all_patches)
    n_val = max(1, int(len(all_patches) * val_ratio))
    val_set = set(all_patches[:n_val])

    # --- build txt lines ---
    train_lines, val_lines = [], []
    train_ref_lines, val_ref_lines = [], []

    for pname in sorted(all_patches):
        img_entry = f"image/{pname}"
        ref_entry = f"image_refined/{pname}"
        msk_entry = f"mask/{pname}"
        if pname in val_set:
            val_lines.append(f"{img_entry},{msk_entry}")
            val_ref_lines.append(f"{ref_entry},{msk_entry}")
        else:
            train_lines.append(f"{img_entry},{msk_entry}")
            train_ref_lines.append(f"{ref_entry},{msk_entry}")

    # --- write split files ---
    for fname, lines in [
        ("train_data.txt", train_lines),
        ("validating_data.txt", val_lines),
        ("train_data_refined.txt", train_ref_lines),
        ("validating_data_refined.txt", val_ref_lines),
    ]:
        (out_dir / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSplit: {len(train_lines)} train patches, {len(val_lines)} val patches "
          f"(ratio {len(val_lines)/len(all_patches):.1%})")
    print(f"Written to {out_dir}:")
    print(f"  train_data.txt              ({len(train_lines)} lines)")
    print(f"  validating_data.txt         ({len(val_lines)} lines)")
    print(f"  train_data_refined.txt      ({len(train_ref_lines)} lines)")
    print(f"  validating_data_refined.txt ({len(val_ref_lines)} lines)")
