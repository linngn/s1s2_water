"""
Normalize a 7-band fused raster (from prepare_combined_dataset.py).

Input band order (as produced by prepare_combined_dataset.py):
  1) Red        – S2 TOA reflectance (DN / 10000), float32
  2) Green      – S2 TOA reflectance (DN / 10000), float32
  3) Blue       – S2 TOA reflectance (DN / 10000), float32
  4) MNDWI      – (Green - SWIR1) / (Green + SWIR1 + eps), range ~ [-1, 1]
  5) VV_db      – S1 backscatter in dB (Lee-filtered), float32
  6) VH_db      – S1 backscatter in dB (Lee-filtered), float32
  7) cloud_mask – binary (0 = no cloud, 1 = cloud)

Output band order (all float32, NoData preserved as NaN):
  1) Red        – reflectance clipped to [0, 1]
  2) Green      – reflectance clipped to [0, 1]
  3) Blue       – reflectance clipped to [0, 1]
  4) MNDWI      – clipped to [0, 1]
  5) VV_db      – dB clipped to [-25, 5],  then min-max scaled to [0, 1]
  6) VH_db      – dB clipped to [-32, -5], then min-max scaled to [0, 1]
  7) cloud_mask – unchanged, [0, 1]

Usage:
  python standardize_raster.py <input_path> <output_path>
"""

import argparse
import numpy as np
import rasterio

# Fixed dB clipping bounds for SAR bands
VV_BOUNDS = (-25.0, 5.0)
VH_BOUNDS = (-32.0, -5.0)


def _valid_mask(data: np.ndarray, nodata) -> np.ndarray:
    """Return boolean mask of valid (non-NoData, non-NaN) pixels."""
    mask = np.ones(data.shape, dtype=bool)
    if nodata is not None:
        mask &= data != nodata
    mask &= ~np.isnan(data)
    return mask


def _clip_to_01(data: np.ndarray, valid: np.ndarray) -> None:
    """Clip valid pixels to [0, 1] in-place. Invalid pixels set to NaN."""
    np.clip(data, 0.0, 1.0, out=data)
    data[~valid] = np.nan


def _clip_and_minmax(data: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> None:
    """Clip valid pixels to [lo, hi], then min-max scale to [0, 1] in-place."""
    data[valid] = np.clip(data[valid], lo, hi)
    data[valid] = (data[valid] - lo) / (hi - lo)
    data[~valid] = np.nan


def normalize_raster(input_path: str, output_path: str) -> None:
    """Normalize each band of the 7-band fused raster and write the result."""
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        nodata = src.nodata
        band_count = src.count

        if band_count != 7:
            raise ValueError(f"Expected 7-band raster, got {band_count} bands")

        profile.update(dtype="float32", nodata=np.nan)

        with rasterio.open(output_path, "w", **profile) as dst:
            for b in range(1, 8):
                data = src.read(b, out_dtype="float32")
                valid = _valid_mask(data, nodata)

                if b in (1, 2, 3):
                    # R, G, B – reflectance already in ~[0, 1], clip to [0, 1]
                    _clip_to_01(data, valid)

                elif b == 4:
                    # MNDWI – clip to [0, 1]
                    _clip_to_01(data, valid)

                elif b == 5:
                    # VV dB – clip to [-25, 5] then scale to [0, 1]
                    _clip_and_minmax(data, valid, *VV_BOUNDS)

                elif b == 6:
                    # VH dB – clip to [-32, -5] then scale to [0, 1]
                    _clip_and_minmax(data, valid, *VH_BOUNDS)

                elif b == 7:
                    # cloud mask – already [0, 1], pass through
                    data[~valid] = np.nan

                dst.write(data, b)

            # preserve band descriptions from source
            for i in range(1, 8):
                desc = src.descriptions[i - 1]
                if desc:
                    dst.set_band_description(i, desc)

    print(f"Normalized raster saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize a 7-band fused raster (RGB, MNDWI, VV, VH, cloud) to [0, 1]."
    )
    parser.add_argument("input_path", help="Path to the input 7-band raster")
    parser.add_argument("output_path", help="Path for the output normalized raster")
    args = parser.parse_args()

    normalize_raster(args.input_path, args.output_path)
