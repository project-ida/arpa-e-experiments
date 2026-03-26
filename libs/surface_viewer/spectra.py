from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import keV_to_channel


def stack_spectra(spec_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Stack spectra to the maximum common length using zero-padding."""
    spec_series = spec_series.dropna()
    spec_series = spec_series[
        spec_series.map(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0)
    ]
    if spec_series.empty:
        raise ValueError("No valid spectra to stack.")

    max_len = int(max(len(s) for s in spec_series))

    def pad_to(s, L):
        a = np.asarray(s, dtype=np.int64)
        if a.size < L:
            a = np.pad(a, (0, L - a.size))
        return a

    stack = np.vstack([pad_to(s, max_len) for s in spec_series])
    x = np.arange(max_len)
    return stack, x


def stack_spectra_trim(spec_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Stack spectra using the minimum common length across spectra."""
    spec_series = spec_series.dropna()
    spec_series = spec_series[
        spec_series.map(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0)
    ]
    if spec_series.empty:
        raise ValueError("No valid spectra to stack.")

    min_len = int(min(len(s) for s in spec_series))
    stack = np.vstack([np.asarray(s[:min_len], dtype=float) for s in spec_series])
    x = np.arange(min_len)
    return stack, x


def band_sum(spectrum, band_start, band_end):
    s = np.asarray(spectrum, dtype=float)
    if s.size == 0:
        return np.nan
    lo = max(0, min(int(band_start), s.size - 1))
    hi = max(0, min(int(band_end), s.size - 1))
    if hi < lo:
        lo, hi = hi, lo
    return float(s[lo:hi + 1].sum())


def summarize_band_values(values):
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "min": np.nan, "p01": np.nan, "p05": np.nan, "p10": np.nan,
            "p25": np.nan, "p50": np.nan, "p75": np.nan, "p90": np.nan,
            "p95": np.nan, "p99": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan,
        }

    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def resolve_band_to_channels(band_start, band_end, band_mode="channels", cal=None):
    mode = str(band_mode).strip().lower()
    if mode == "channels":
        lo = int(round(float(band_start)))
        hi = int(round(float(band_end)))
    elif mode == "kev":
        if cal is None:
            raise ValueError("Energy calibration is required when BAND_MODE='keV'")
        lo = keV_to_channel(band_start, cal)
        hi = keV_to_channel(band_end, cal)
    else:
        raise ValueError(f"Unknown BAND_MODE: {band_mode}")
    return (lo, hi) if lo <= hi else (hi, lo)


def band_label_text(band_start, band_end, band_mode="channels") -> str:
    mode = str(band_mode).strip().lower()
    if mode == "kev":
        return f"Band-sum ({band_start}–{band_end} keV)"
    return f"Band-sum ({band_start}–{band_end} channels)"


def print_cli_suggestions(
    stats_df,
    roi_name,
    *,
    band_start,
    band_end,
    input_folder_placeholder="aggregated-spectra",
    script_placeholder="create_heatmap_overlay.py",
):
    """Print example CLI commands for creating heatmap overlays from summary stats."""
    row = stats_df.loc[roi_name]

    vmin_mild = int(round(row["p05"]))
    vmax_mild = int(round(row["p95"]))
    vmin_strong = int(round(row["p10"]))
    vmax_strong = int(round(row["p90"]))
    threshold = int(round(row["p10"]))

    print(f"\n=== {roi_name} ===")
    print("Suggested milder robust scaling:")
    print(
        f"python {script_placeholder} "
        f'--input-folder "{input_folder_placeholder}" '
        f"--band-start {band_start} --band-end {band_end} "
        f"--vmin {vmin_mild} --vmax {vmax_mild}"
    )

    print("Suggested stronger clipping:")
    print(
        f"python {script_placeholder} "
        f'--input-folder "{input_folder_placeholder}" '
        f"--band-start {band_start} --band-end {band_end} "
        f"--vmin {vmin_strong} --vmax {vmax_strong}"
    )

    print("Possible threshold choice:")
    print(
        f"python {script_placeholder} "
        f'--input-folder "{input_folder_placeholder}" '
        f"--band-start {band_start} --band-end {band_end} "
        f"--threshold {threshold}"
    )

    print("Threshold + scaling together:")
    print(
        f"python {script_placeholder} "
        f'--input-folder "{input_folder_placeholder}" '
        f"--band-start {band_start} --band-end {band_end} "
        f"--vmin {vmin_strong} --vmax {vmax_strong} "
        f"--threshold {threshold}"
    )
