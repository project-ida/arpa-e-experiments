from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.sparse.linalg import spsolve


def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """Asymmetric Least Squares baseline."""
    y = np.asarray(y, dtype=float)
    L = y.size
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess(y, smooth_window=21, smooth_poly=3, do_baseline=True, lam=1e6, p=0.01):
    """Baseline-correct and smooth the signal."""
    y = np.asarray(y, dtype=float)

    if smooth_window and smooth_window > 3:
        w = min(int(smooth_window) | 1, len(y) - (1 - len(y) % 2))
        w = max(5, w | 1)
        y_s = savgol_filter(y, window_length=w, polyorder=min(smooth_poly, w - 1))
    else:
        y_s = y

    if do_baseline:
        b = baseline_als(y_s, lam=lam, p=p)
        y_corr = np.clip(y_s - b, 0, None)
    else:
        y_corr = y_s

    return y_corr


def estimate_noise(y):
    """Robust noise estimate using MAD of first differences."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return 0.0
    d = np.diff(y)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    return 1.4826 * mad


def detect_peaks(
    y,
    x=None,
    min_prom=None,
    min_height=None,
    min_distance=5,
    rel_height=0.5,
    max_peaks=None,
):
    """Return ``(peaks_df, meta)`` with x, idx, height, prominence, fwhm, and area."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y)) if x is None else np.asarray(x, dtype=float)

    yc = preprocess(y)
    noise = estimate_noise(yc)

    if min_prom is None:
        min_prom = 6 * noise
    if min_height is None:
        min_height = 3 * noise

    idx, props = find_peaks(yc, prominence=min_prom, height=min_height, distance=min_distance)

    if idx.size == 0:
        df = pd.DataFrame(columns=["x", "idx", "height", "prominence", "fwhm", "area"])
        return df, {"noise": noise, "min_prom": min_prom, "min_height": min_height}

    results = peak_widths(yc, idx, rel_height=rel_height)
    widths = results[0]
    left_ips, right_ips = results[2], results[3]

    areas = []
    for li, ri in zip(left_ips, right_ips):
        li_i = max(0, int(np.floor(li)))
        ri_i = min(len(yc) - 1, int(np.ceil(ri)))
        areas.append(float(np.trapz(yc[li_i:ri_i + 1], x[li_i:ri_i + 1])))

    df = pd.DataFrame({
        "x": x[idx],
        "idx": idx,
        "height": props.get("peak_heights", np.zeros_like(idx, dtype=float)),
        "prominence": props.get("prominences", np.zeros_like(idx, dtype=float)),
        "fwhm": widths,
        "area": areas,
    }).sort_values("height", ascending=False)

    if max_peaks:
        df = df.head(int(max_peaks)).sort_values("x").reset_index(drop=True)
    else:
        df = df.sort_values("x").reset_index(drop=True)

    return df, {"noise": noise, "min_prom": min_prom, "min_height": min_height}


def plot_with_peaks(y, x=None, peaks_df=None, title="Spectrum with detected peaks", xlabel="Channel"):
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float)
    x = np.arange(len(y)) if x is None else np.asarray(x, dtype=float)

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, y, label="signal")

    if peaks_df is not None and not peaks_df.empty:
        plt.scatter(peaks_df["x"], peaks_df["height"], marker="x", label="peaks")

    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title)
    plt.tight_layout()
    plt.show()


_COMMON_LINES_KEV = {
    "B": {"Ka1": 0.1833},
    "C": {"Ka1": 0.277},
    "N": {"Ka1": 0.3924},
    "O": {"Ka1": 0.5249},
    "F": {"Ka1": 0.6768},
    "Na": {"Ka1": 1.0403},
    "Mg": {"Ka1": 1.25379, "Kb1": 1.302},
    "Al": {"Ka1": 1.4865, "Kb1": 1.557},
    "Si": {"Ka1": 1.7398, "Kb1": 1.837},
    "P": {"Ka1": 2.0105, "Kb1": 2.1395},
    "S": {"Ka1": 2.3095, "Kb1": 2.465},
    "Cl": {"Ka1": 2.622, "Kb1": 2.812},
    "K": {"Ka1": 3.3138, "Kb1": 3.5901},
    "Ca": {"Ka1": 3.6923, "Kb1": 4.0131},
    "Ti": {"Ka1": 4.5122, "Kb1": 4.9334, "La1": 0.4518, "Lb1": 0.4582},
    "Cr": {"Ka1": 5.4149, "Kb1": 5.9468, "La1": 0.5721, "Lb1": 0.5818},
    "Mn": {"Ka1": 5.9003, "Kb1": 6.4918, "La1": 0.6367, "Lb1": 0.6479},
    "Fe": {"Ka1": 6.4052, "Kb1": 7.0593, "La1": 0.7048, "Lb1": 0.7179},
    "Co": {"Ka1": 6.9309, "Kb1": 7.6491, "La1": 0.7751, "Lb1": 0.7902},
    "Ni": {"Ka1": 7.4803, "Kb1": 8.2668, "La1": 0.8487, "Lb1": 0.866},
    "Cu": {"Ka1": 8.0463, "Kb1": 8.9039, "La1": 0.9277, "Lb1": 0.9473},
    "Zn": {"Ka1": 8.6372, "Kb1": 9.5704, "La1": 1.0117, "Lb1": 1.0347},
    "Zr": {"Ka1": 15.775, "Kb1": 17.6682, "La1": 2.0442, "Lb1": 2.1259},
    "Nb": {"Ka1": 16.615, "Kb1": 18.6254, "La1": 2.1687, "Lb1": 2.26},
    "Mo": {"Ka1": 17.48, "Kb1": 19.606, "La1": 2.2921, "Lb1": 2.3939},
    "Pd": {"La1": 2.8378, "Lb1": 2.9895},
    "Ag": {"La1": 2.9827, "Lb1": 3.15},
    "Sn": {"La1": 3.4441, "Lb1": 3.6628},
    "Ta": {"La1": 8.146, "Lb1": 9.343},
    "W": {"La1": 8.398, "Lb1": 9.672},
    "Pt": {"La1": 9.442, "Lb1": 11.071},
    "Au": {"La1": 9.713, "Lb1": 11.443},
}


def line_library(beam_keV=15.0, elements=None, include=("Ka1", "Kb1", "La1", "Lb1")) -> pd.DataFrame:
    """Build a line-library DataFrame from the minimal built-in X-ray line list."""
    rows = []
    use = elements if elements else _COMMON_LINES_KEV.keys()

    for el in use:
        d = _COMMON_LINES_KEV.get(el, {})
        for line, E in d.items():
            if include and line not in include:
                continue
            E = float(E)
            if E <= beam_keV:
                rows.append({
                    "element": el,
                    "line": line,
                    "energy_keV": E,
                    "E_keV": E,
                    "label": f"{el} {line}",
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("energy_keV", ignore_index=True)


def _fwhm_model_const(fwhm_eV=67.8):
    f = float(fwhm_eV) / 1000.0
    return lambda E_keV: f


def _match_peaks_to_lines(
    peaks_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    *,
    sigma: float = 2.5,
    fwhm_func=None,
) -> pd.DataFrame:
    """Match each peak to the nearest library line within ``sigma * σ(E)``."""
    if peaks_df is None or peaks_df.empty or lines_df is None or lines_df.empty:
        return pd.DataFrame(columns=[
            "energy_keV", "idx", "height", "prominence", "area", "fwhm",
            "element", "line", "lib_energy_keV", "delta_keV", "label",
        ])

    energy_col = "energy_keV" if "energy_keV" in lines_df.columns else "E_keV"
    if energy_col not in lines_df.columns:
        raise KeyError("lines_df must have 'energy_keV' or 'E_keV'.")

    if fwhm_func is None:
        fwhm_func = _fwhm_model_const(67.8)

    lib_E = lines_df[energy_col].to_numpy()
    out = []

    for _, p in peaks_df.iterrows():
        e = float(p["x"])
        fwhm = float(fwhm_func(e))
        sigma_E = fwhm / 2.355 if fwhm > 0 else 0.05

        j = int(np.argmin(np.abs(lib_E - e)))
        delta = float(lib_E[j] - e)

        if abs(delta) <= sigma * sigma_E:
            r = lines_df.iloc[j]
            lib_e = float(r[energy_col])
            out.append({
                "energy_keV": e,
                "idx": int(p.get("idx", -1)),
                "height": float(p.get("height", np.nan)),
                "prominence": float(p.get("prominence", np.nan)),
                "area": float(p.get("area", np.nan)),
                "fwhm": float(p.get("fwhm", np.nan)),
                "element": r.get("element", ""),
                "line": r.get("line", ""),
                "lib_energy_keV": lib_e,
                "delta_keV": delta,
                "label": r.get("label", f"{r.get('element', '')} {r.get('line', '')}".strip()),
            })

    return pd.DataFrame(out).sort_values("energy_keV", ignore_index=True)


def identify_elements(
    cum,
    *,
    x_keV=None,
    eV_per_ch=20.000347,
    start_eV=-192.768,
    beam_keV=15.0,
    fwhm_Mn_eV=67.8,
    elements=None,
    include=("Ka1", "Kb1", "La1", "Lb1"),
    max_peaks=30,
    sigma=2.5,
    min_distance=5,
    custom_lines_df=None,
):
    """Identify likely element lines in an aggregate EDS spectrum."""
    cum = np.asarray(cum)

    if x_keV is None:
        L = len(cum)
        x_keV = (start_eV + np.arange(L) * eV_per_ch) / 1000.0
    else:
        x_keV = np.asarray(x_keV)

    if custom_lines_df is not None:
        lines_df = custom_lines_df.copy()
        if "energy_keV" not in lines_df.columns and "E_keV" in lines_df.columns:
            lines_df = lines_df.rename(columns={"E_keV": "energy_keV"})
        if "E_keV" not in lines_df.columns and "energy_keV" in lines_df.columns:
            lines_df["E_keV"] = lines_df["energy_keV"]
        if "label" not in lines_df.columns:
            lines_df["label"] = lines_df["element"].astype(str) + " " + lines_df["line"].astype(str)
        lines_df = lines_df[lines_df["energy_keV"] <= beam_keV].sort_values("energy_keV", ignore_index=True)
    else:
        lines_df = line_library(beam_keV=beam_keV, elements=elements, include=include)

    peaks_df, meta = detect_peaks(cum, x=x_keV, max_peaks=max_peaks, min_distance=min_distance)

    fwhm_func = _fwhm_model_const(fwhm_Mn_eV)
    assign_df = _match_peaks_to_lines(peaks_df, lines_df, sigma=sigma, fwhm_func=fwhm_func)

    return assign_df, peaks_df, lines_df, meta
