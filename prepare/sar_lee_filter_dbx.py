"""
sar_lee_filter_dbx.py

Process a preprocessed Sentinel-1 GeoTIFF where values are stored as:
- Int16
- dB scaled by factor (default 100): stored_value = dB * 100

Pipeline:
1) Read band from GeoTIFF
2) Convert stored dB*x -> true dB -> linear
3) Apply:
   - Basic Lee filter (classic)
   - Refined Lee filter (directional approximation)
4) Save outputs (default: dB*x Int16, matching the dataset format)

Install:
  pip install rasterio numpy scipy

Examples:
  python sar_lee_filter_dbx.py --in "D:\\...\\sentinel12_s1_71_img.tif" --out_dir outputs --band 2
  python sar_lee_filter_dbx.py --in "D:\\...\\sentinel12_s1_71_img.tif" --out_dir outputs --band 2 --db_scale 100 --save_format int16_dbx
  python sar_lee_filter_dbx.py --in "D:\\...\\sentinel12_s1_71_img.tif" --out_dir outputs --band 2 --save_format float_db
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import convolve

EPS = 1e-12


def _ensure_odd(ksize: int) -> int:
    if ksize < 1:
        raise ValueError("ksize must be >= 1")
    return ksize if (ksize % 2 == 1) else (ksize + 1)


def _valid_mask(img: np.ndarray, nodata: float | None) -> np.ndarray:
    m = np.isfinite(img)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        m &= (img != nodata)
    return m


def db_to_linear_safe(db: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    dB -> linear, with clipping to avoid overflow if anything weird slips in.
    Sentinel-1 backscatter is typically around [-35, +5] dB. We clip wider for safety.
    """
    db64 = db.astype(np.float64, copy=False)
    db64 = np.clip(db64, -80.0, 30.0)
    out = np.full(db.shape, np.nan, dtype=np.float32)
    out[valid] = np.power(10.0, db64[valid] / 10.0).astype(np.float32)
    return out


def linear_to_db_safe(lin: np.ndarray) -> np.ndarray:
    lin64 = lin.astype(np.float64, copy=False)
    return (10.0 * np.log10(np.maximum(lin64, EPS))).astype(np.float32)


def lee_filter_basic(
    img_linear: np.ndarray,
    *,
    ksize: int = 7,
    enl: float = 4.0,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Classic Lee filter (linear domain):
      out = mean + W*(img - mean)
      W = clip((cv^2 - 1/ENL) / (cv^2 + eps), 0, 1)
      cv^2 = var / mean^2
    """
    if enl <= 0:
        raise ValueError("enl must be > 0")
    ksize = _ensure_odd(ksize)

    img = img_linear.astype(np.float32, copy=True)

    # valid pixels: finite + not nodata + non-negative
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        valid = np.isfinite(img) & (img >= 0)
    else:
        valid = np.isfinite(img) & (img != nodata) & (img >= 0)

    img[~valid] = np.nan

    k = np.ones((ksize, ksize), dtype=np.float32)
    w = np.isfinite(img).astype(np.float32)

    sum_x = convolve(np.nan_to_num(img, nan=0.0), k, mode="nearest")

    img0 = np.nan_to_num(img, nan=0.0).astype(np.float64)
    sum_x2 = convolve(img0 * img0, k, mode="nearest")

    cnt = convolve(w, k, mode="nearest")
    cnt = np.maximum(cnt, 1.0)

    mean = (sum_x / cnt).astype(np.float64)
    var = np.maximum(sum_x2 / cnt - mean * mean, 0.0)

    cv2 = var / np.maximum(mean * mean, EPS)
    nv = 1.0 / float(enl)
    W = np.clip((cv2 - nv) / (cv2 + EPS), 0.0, 1.0).astype(np.float32)

    mean32 = mean.astype(np.float32)
    out = mean32 + W * (img - mean32)
    out = out.astype(np.float32, copy=False)

    # restore invalid
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        out[~valid] = np.nan
    else:
        out[~valid] = nodata
        out[~np.isfinite(out)] = nodata

    return out


def lee_filter_refined(
    img_linear: np.ndarray,
    *,
    ksize: int = 7,
    enl: float = 4.0,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Refined Lee (directional approximation):
    - Use square-window mean for direction detection
    - Choose direction by max gradient on mean
    - Compute directional mean/var using line masks
    - Apply Lee formula using directional stats
    """
    if enl <= 0:
        raise ValueError("enl must be > 0")
    ksize = _ensure_odd(ksize)
    r = ksize // 2

    img = img_linear.astype(np.float32, copy=True)

    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        valid = np.isfinite(img) & (img >= 0)
    else:
        valid = np.isfinite(img) & (img != nodata) & (img >= 0)

    img[~valid] = np.nan

    # square stats for direction detection
    k_sq = np.ones((ksize, ksize), dtype=np.float32)
    w = np.isfinite(img).astype(np.float32)

    sum_x = convolve(np.nan_to_num(img, nan=0.0), k_sq, mode="nearest")
    cnt = np.maximum(convolve(w, k_sq, mode="nearest"), 1.0)
    mean_sq = (sum_x / cnt).astype(np.float32)

    # gradient kernels on mean_sq
    kh = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float32)  # E-W
    kv = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)  # N-S
    kd1 = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32)  # NW-SE
    kd2 = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], dtype=np.float32)  # NE-SW

    mean_for_grad = np.nan_to_num(mean_sq, nan=0.0).astype(np.float32)
    gh = np.abs(convolve(mean_for_grad, kh, mode="nearest"))
    gv = np.abs(convolve(mean_for_grad, kv, mode="nearest"))
    gd1 = np.abs(convolve(mean_for_grad, kd1, mode="nearest"))
    gd2 = np.abs(convolve(mean_for_grad, kd2, mode="nearest"))

    grads = np.stack([gh, gv, gd1, gd2], axis=0)
    dir_idx = np.argmax(grads, axis=0).astype(np.uint8)

    # directional line masks
    mask_h = np.zeros((ksize, ksize), dtype=np.float32); mask_h[r, :] = 1.0
    mask_v = np.zeros((ksize, ksize), dtype=np.float32); mask_v[:, r] = 1.0
    mask_d1 = np.zeros((ksize, ksize), dtype=np.float32); np.fill_diagonal(mask_d1, 1.0)
    mask_d2 = np.zeros((ksize, ksize), dtype=np.float32); np.fill_diagonal(np.fliplr(mask_d2), 1.0
    )
    masks = [mask_h, mask_v, mask_d1, mask_d2]

    img0 = np.nan_to_num(img, nan=0.0).astype(np.float64)
    img02 = img0 * img0
    w0 = np.isfinite(img).astype(np.float32)

    out = np.full(img.shape, np.nan, dtype=np.float32)
    nv = 1.0 / float(enl)

    for d, m in enumerate(masks):
        sum_d = convolve(img0, m, mode="nearest")
        sum2_d = convolve(img02, m, mode="nearest")
        cnt_d = np.maximum(convolve(w0, m, mode="nearest"), 1.0)

        mean_d = (sum_d / cnt_d).astype(np.float64)
        var_d = np.maximum(sum2_d / cnt_d - mean_d * mean_d, 0.0)

        cv2_d = var_d / np.maximum(mean_d * mean_d, EPS)
        W_d = np.clip((cv2_d - nv) / (cv2_d + EPS), 0.0, 1.0).astype(np.float32)

        mean_d32 = mean_d.astype(np.float32)
        out_d = mean_d32 + W_d * (img - mean_d32)

        sel = (dir_idx == d) & valid
        out[sel] = out_d[sel].astype(np.float32, copy=False)

    # fallback to basic where needed
    still = valid & (~np.isfinite(out))
    if np.any(still):
        out_basic = lee_filter_basic(img_linear, ksize=ksize, enl=enl, nodata=nodata)
        out[still] = out_basic[still]

    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        out[~valid] = np.nan
    else:
        out[~valid] = nodata
        out[~np.isfinite(out)] = nodata

    return out


def _write_geotiff(out_path: Path, arr: np.ndarray, profile: dict, nodata: float | None) -> None:
    prof = profile.copy()
    prof.update(count=1, compress="deflate", tiled=True, BIGTIFF="IF_SAFER")

    if arr.dtype == np.float32:
        prof.update(dtype="float32", predictor=3)
        # For float outputs, it's usually best to omit nodata if it's not meaningful.
        # But we keep it if the input had one and it's not NaN.
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            prof["nodata"] = float(nodata)
        else:
            prof.pop("nodata", None)
    elif arr.dtype == np.int16:
        prof.update(dtype="int16")
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            prof["nodata"] = int(nodata)
        else:
            prof.pop("nodata", None)
    else:
        raise ValueError(f"Unsupported dtype for writing: {arr.dtype}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input GeoTIFF path (preprocessed S1)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--band", type=int, default=1, help="1-based band index (0:VV, 1:VH in your dataset)")
    ap.add_argument("--ksize", type=int, default=7, help="Kernel/window size (odd preferred)")
    ap.add_argument("--enl", type=float, default=4.0, help="Equivalent Number of Looks")
    ap.add_argument("--db_scale", type=float, default=100.0,
                    help="Scale factor of stored dB (e.g., stored = dB*100 -> db_scale=100)")
    ap.add_argument("--save_format", choices=["int16_dbx", "float_db", "float_linear"], default="int16_dbx",
                    help="Output format: int16_dbx saves dB*db_scale int16; float_db saves float32 dB; float_linear saves float32 linear")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    ksize = _ensure_odd(args.ksize)

    with rasterio.open(in_path) as src:
        profile = src.profile
        nodata = src.nodata
        raw = src.read(args.band, out_dtype="float32")  # read as float for math

    valid = _valid_mask(raw, nodata)

    # 1) stored (dB*db_scale) -> true dB
    db = np.full(raw.shape, np.nan, dtype=np.float32)
    db[valid] = raw[valid] / float(args.db_scale)

    # 2) dB -> linear for Lee
    lin = db_to_linear_safe(db, valid)

    # 3) Lee filters in linear
    lin_basic = lee_filter_basic(lin, ksize=ksize, enl=args.enl, nodata=np.nan)
    lin_ref = lee_filter_refined(lin, ksize=ksize, enl=args.enl, nodata=np.nan)

    stem = in_path.stem

    # 4) Save
    if args.save_format == "float_linear":
        out_basic = lin_basic.astype(np.float32)
        out_ref = lin_ref.astype(np.float32)
        out_basic_path = out_dir / f"{stem}_lee_basic_ks{ksize}_enl{args.enl:g}_linear.tif"
        out_ref_path = out_dir / f"{stem}_lee_refined_ks{ksize}_enl{args.enl:g}_linear.tif"
        _write_geotiff(out_basic_path, out_basic, profile, nodata=None)
        _write_geotiff(out_ref_path, out_ref, profile, nodata=None)

    else:
        db_basic = linear_to_db_safe(lin_basic)
        db_ref = linear_to_db_safe(lin_ref)

        if args.save_format == "float_db":
            out_basic = db_basic.astype(np.float32)
            out_ref = db_ref.astype(np.float32)
            out_basic_path = out_dir / f"{stem}_lee_basic_ks{ksize}_enl{args.enl:g}_db.tif"
            out_ref_path = out_dir / f"{stem}_lee_refined_ks{ksize}_enl{args.enl:g}_db.tif"
            _write_geotiff(out_basic_path, out_basic, profile, nodata=None)
            _write_geotiff(out_ref_path, out_ref, profile, nodata=None)

        else:  # int16_dbx
            # dB -> stored int16 (dB*db_scale)
            out_basic = np.round(db_basic * float(args.db_scale)).astype(np.int16)
            out_ref = np.round(db_ref * float(args.db_scale)).astype(np.int16)

            # restore nodata
            if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                out_basic[~valid] = int(nodata)
                out_ref[~valid] = int(nodata)

            out_basic_path = out_dir / f"{stem}_lee_basic_ks{ksize}_enl{args.enl:g}_dbx{int(args.db_scale)}.tif"
            out_ref_path = out_dir / f"{stem}_lee_refined_ks{ksize}_enl{args.enl:g}_dbx{int(args.db_scale)}.tif"
            _write_geotiff(out_basic_path, out_basic, profile, nodata=nodata)
            _write_geotiff(out_ref_path, out_ref, profile, nodata=nodata)

    print("Saved:")
    print(" -", out_basic_path)
    print(" -", out_ref_path)


if __name__ == "__main__":
    main()