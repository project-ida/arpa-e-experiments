---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="c9b739d7-5903-4368-867e-16b9d7136692" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/templates/eds-spectrum-starter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/templates/eds-spectrum-starter.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="7f188c81" -->
# EDS Spectrum template

This notebook provides a quick, end-to-end demo for ROI-based EDS analysis:
it pulls a saved ROI set from the Surface Viewer API, builds a table of selected tiles,
loads the corresponding spectra, aggregates them, and runs a simple first-pass peak analysis.

## How to use

1. Add the `api_url` (get the link from a surface viewer selection grid)
name.
2. Run cells top-to-bottom.
3. Use the aggregate plot and peak table as a rapid “first look” before moving to the full testbed.

<!-- #endregion -->

```python id="Z5okfpEzLR6J"
api_url = 'CHANGE_THIS'

```

```python id="1b17d9ed"

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from urllib.parse import urlparse, parse_qs, urljoin, quote

import re
import requests

from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter


# -----------------------------
# ROI loading from the API JSON
# -----------------------------

def load_roi_api(api_url: str) -> pd.DataFrame:
    """
    Fetch ROI selections from:
        .../api/rois.php?dataset=<dataset>&name=<roi_name>

    Expected response schema:
        { "selections": [{col,row,srcJson,basename,foldername}, ...] }

    Returns a DataFrame with at least:
        col, row, srcJson, basename, foldername
    """
    r = requests.get(api_url, timeout=60)
    r.raise_for_status()
    data = r.json()
    selections = data.get("selections", [])
    df = pd.DataFrame(selections)
    if df.empty:
        return df

    # types
    for c in ["row", "col"]:
        if c in df.columns:
            df[c] = df[c].astype("int64")

    # normalise expected string columns if present
    for c in ["srcJson", "basename", "foldername"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def infer_dataset_base_from_api(api_url: str) -> str:
    """
    Try to infer the dataset base URL for spectrum JSON files based on the
    'dataset' query parameter used by the ROI API.
    """
    p = urlparse(api_url)
    qs = parse_qs(p.query)
    dataset = (qs.get("dataset") or [None])[0]
    if not dataset:
        raise ValueError("Could not find 'dataset' parameter in the API URL.")

    # Root pattern used by Surface Viewer
    root = f"{p.scheme}://{p.netloc}/surface-viewer/data/"

    # URL-encode the dataset name to form a safe folder path
    dataset_encoded = quote(dataset, safe="")
    return urljoin(root, dataset_encoded + "/")


def add_json_urls(df: pd.DataFrame, api_url: str) -> pd.DataFrame:
    """
    Add a json_url column by resolving srcJson against the inferred dataset base.
    """
    if df.empty:
        df["json_url"] = pd.Series(dtype="string")
        return df

    dataset_base = infer_dataset_base_from_api(api_url)
    df = df.copy()
    df["json_url"] = df["srcJson"].apply(lambda p: urljoin(dataset_base, str(p)))
    return df


# -----------------------------------------
# Minimal aggregation helpers (from testbed)
# -----------------------------------------

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "eds-demo-notebook/0.1"})
    return s


def fetch_json_items(url: str, session: requests.Session) -> list:
    """
    GET a JSON file and return a list of records.
    Supports files that are either:
        - an array of records, or
        - { "items": [ ... ] }.
    """
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def build_spectrum_index(urls: list[str], *, progress: bool = True,
                         session: requests.Session | None = None) -> dict[tuple[str,int,int], list[int]]:
    """
    For each JSON URL, read it once and build a lookup:
        (json_url, row, col) -> [counts...]
    Accepts row/col under 'row'/'col' or 'rownum'/'colnum'.
    Accepts spectrum under 'aggregatedspectrum' / 'aggregatedSpectrum' / 'spectrum'.
    """
    session = session or _new_session()
    index: dict[tuple[str,int,int], list[int]] = {}

    for url in tqdm(urls, desc="Downloading JSON files", disable=not progress):
        items = fetch_json_items(url, session)
        for rec in items:
            r = rec.get("rownum", rec.get("row"))
            c = rec.get("colnum", rec.get("col"))
            spec = rec.get("aggregatedspectrum") or rec.get("aggregatedSpectrum") or rec.get("spectrum")
            if r is None or c is None or spec is None:
                continue
            try:
                key = (url, int(r), int(c))
                index[key] = [int(x) for x in spec]
            except Exception:
                pass
    return index


def attach_spectra(df: pd.DataFrame,
                   index: dict[tuple[str,int,int], list[int]],
                   *, progress: bool = True) -> pd.DataFrame:
    """Add a 'spectrum' column to df using the (url,row,col)->spectrum index."""
    def pick(row):
        return index.get((row["json_url"], int(row["row"]), int(row["col"])), None)

    df = df.copy()
    if progress:
        tqdm.pandas(desc="Indexing spectra")
        df["spectrum"] = df.progress_apply(pick, axis=1)
    else:
        df["spectrum"] = df.apply(pick, axis=1)
    return df


def stack_spectra(spec_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a Series of list-like spectra, returns:
        - stack: (n_spectra x max_len) array, zero-padded
        - x: channel indices
    """
    spec_series = spec_series.dropna()
    spec_series = spec_series[spec_series.map(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)]
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


def plot_cumulative(stack: np.ndarray, x: np.ndarray, *, title_prefix="Cumulative spectrum"):
    cum = stack.sum(axis=0)
    n = stack.shape[0]
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, cum)
    plt.xlabel("Channel")
    plt.ylabel("Counts")
    plt.title(f"{title_prefix} (n={n})")
    plt.tight_layout()
    plt.show()
    return cum


def plot_overlay(stack: np.ndarray, x: np.ndarray, *, max_traces=50, title="Overlay spectra"):
    n = stack.shape[0]
    plt.figure(figsize=(10, 4.5))
    step = max(1, n // max_traces)
    for i in range(0, n, step):
        plt.plot(x, stack[i], alpha=0.25)
    plt.xlabel("Channel")
    plt.ylabel("Counts")
    plt.title(f"{title} (showing ~{min(n, max_traces)} of {n})")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Simple peak detection helpers
# -----------------------------

def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """Asymmetric Least Squares baseline."""
    y = np.asarray(y, dtype=float)
    L = y.size
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
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


def detect_peaks(y, x=None, min_prom=None, min_height=None, min_distance=5,
                 rel_height=0.5, max_peaks=None):
    """
    Return (peaks_df, meta) where peaks_df has:
        x, idx, height, prominence, fwhm, area
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y)) if x is None else np.asarray(x, dtype=float)

    yc = preprocess(y)
    noise = estimate_noise(yc)

    if min_prom is None:
        min_prom = 6 * noise
    if min_height is None:
        min_height = 3 * noise

    idx, props = find_peaks(
        yc,
        prominence=min_prom,
        height=min_height,
        distance=min_distance
    )

    if idx.size == 0:
        df = pd.DataFrame(columns=["x", "idx", "height", "prominence", "fwhm", "area"])
        return df, {"noise": noise, "min_prom": min_prom, "min_height": min_height}

    results = peak_widths(yc, idx, rel_height=rel_height)
    widths = results[0]
    left_ips, right_ips = results[2], results[3]

    # simple trapezoid area under the corrected curve
    areas = []
    for li, ri in zip(left_ips, right_ips):
        li_i = max(0, int(np.floor(li)))
        ri_i = min(len(yc) - 1, int(np.ceil(ri)))
        areas.append(float(np.trapz(yc[li_i:ri_i+1], x[li_i:ri_i+1])))

    fwhm = widths

    df = pd.DataFrame({
        "x": x[idx],
        "idx": idx,
        "height": props.get("peak_heights", np.zeros_like(idx, dtype=float)),
        "prominence": props.get("prominences", np.zeros_like(idx, dtype=float)),
        "fwhm": fwhm,
        "area": areas,
    }).sort_values("height", ascending=False)

    if max_peaks:
        df = df.head(int(max_peaks)).sort_values("x").reset_index(drop=True)
    else:
        df = df.sort_values("x").reset_index(drop=True)

    return df, {"noise": noise, "min_prom": min_prom, "min_height": min_height}


def plot_with_peaks(y, x=None, peaks_df=None, title="Spectrum with detected peaks"):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y)) if x is None else np.asarray(x, dtype=float)

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, y, label="signal")

    if peaks_df is not None and not peaks_df.empty:
        plt.scatter(peaks_df["x"], peaks_df["height"], marker="x", label="peaks")

    plt.xlabel("Channel")
    plt.ylabel("Counts")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Minimal X-ray line library (keV)
# Extend this list as needed.
# -----------------------------
_COMMON_LINES_KEV = {
    "B":  {"Ka1": 0.1833},
    "C":  {"Ka1": 0.277},
    "N":  {"Ka1": 0.3924},
    "O":  {"Ka1": 0.5249},
    "F":  {"Ka1": 0.6768},
    "Na": {"Ka1": 1.0403},
    "Mg": {"Ka1": 1.25379, "Kb1": 1.302},
    "Al": {"Ka1": 1.4865,  "Kb1": 1.557},
    "Si": {"Ka1": 1.7398,  "Kb1": 1.837},
    "P":  {"Ka1": 2.0105,  "Kb1": 2.1395},
    "S":  {"Ka1": 2.3095,  "Kb1": 2.465},
    "Cl": {"Ka1": 2.622,   "Kb1": 2.812},
    "K":  {"Ka1": 3.3138,  "Kb1": 3.5901},
    "Ca": {"Ka1": 3.6923,  "Kb1": 4.0131},

    "Ti": {"Ka1": 4.5122,  "Kb1": 4.9334, "La1": 0.4518, "Lb1": 0.4582},
    "Cr": {"Ka1": 5.4149,  "Kb1": 5.9468, "La1": 0.5721, "Lb1": 0.5818},
    "Mn": {"Ka1": 5.9003,  "Kb1": 6.4918, "La1": 0.6367, "Lb1": 0.6479},
    "Fe": {"Ka1": 6.4052,  "Kb1": 7.0593, "La1": 0.7048, "Lb1": 0.7179},
    "Co": {"Ka1": 6.9309,  "Kb1": 7.6491, "La1": 0.7751, "Lb1": 0.7902},
    "Ni": {"Ka1": 7.4803,  "Kb1": 8.2668, "La1": 0.8487, "Lb1": 0.866},
    "Cu": {"Ka1": 8.0463,  "Kb1": 8.9039, "La1": 0.9277, "Lb1": 0.9473},
    "Zn": {"Ka1": 8.6372,  "Kb1": 9.5704, "La1": 1.0117, "Lb1": 1.0347},

    "Zr": {"Ka1": 15.775,  "Kb1": 17.6682, "La1": 2.0442, "Lb1": 2.1259},
    "Nb": {"Ka1": 16.615,  "Kb1": 18.6254, "La1": 2.1687, "Lb1": 2.26},
    "Mo": {"Ka1": 17.48,   "Kb1": 19.606,  "La1": 2.2921, "Lb1": 2.3939},

    "Pd": {"La1": 2.8378, "Lb1": 2.9895},
    "Ag": {"La1": 2.9827, "Lb1": 3.15},
    "Sn": {"La1": 3.4441, "Lb1": 3.6628},

    "Ta": {"La1": 8.146,  "Lb1": 9.343},
    "W":  {"La1": 8.398,  "Lb1": 9.672},
    "Pt": {"La1": 9.442,  "Lb1": 11.071},
    "Au": {"La1": 9.713,  "Lb1": 11.443},
}


def line_library(beam_keV=15.0, elements=None, include=("Ka1","Kb1","La1","Lb1")) -> pd.DataFrame:
    """
    Build a line library DataFrame from _COMMON_LINES_KEV.
    Returns both 'energy_keV' and 'E_keV' columns for compatibility.
    """
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
                    "E_keV": E,  # alias for convenience
                    "label": f"{el} {line}"
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("energy_keV", ignore_index=True)


def _fwhm_model_const(fwhm_eV=67.8):
    f = float(fwhm_eV) / 1000.0  # keV
    return lambda E_keV: f


def _match_peaks_to_lines(peaks_df: pd.DataFrame,
                          lines_df: pd.DataFrame,
                          *,
                          sigma: float = 2.5,
                          fwhm_func=None) -> pd.DataFrame:
    """
    Match each peak to nearest library line within sigma * σ(E).
    Accepts either 'energy_keV' or 'E_keV' in lines_df.
    """
    if peaks_df is None or peaks_df.empty or lines_df is None or lines_df.empty:
        return pd.DataFrame(columns=[
            "energy_keV","idx","height","prominence","area","fwhm",
            "element","line","lib_energy_keV","delta_keV","label"
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
                "label": r.get("label", f"{r.get('element','')} {r.get('line','')}".strip()),
            })

    return pd.DataFrame(out).sort_values("energy_keV", ignore_index=True)


def identify_elements(cum,
                      *,
                      x_keV=None,
                      eV_per_ch=20.000347,
                      start_eV=-192.768,
                      beam_keV=15.0,
                      fwhm_Mn_eV=67.8,
                      elements=None,
                      include=("Ka1","Kb1","La1","Lb1"),
                      max_peaks=30,
                      sigma=2.5,
                      min_distance=5,
                      # Optional: let you pass a custom lines_df directly
                      custom_lines_df=None):
    """
    Identify likely element lines in an aggregate EDS spectrum.

    Returns:
        assign_df, peaks_df, lines_df, meta
    """
    if "detect_peaks" not in globals():
        raise RuntimeError("identify_elements needs detect_peaks defined earlier in the notebook.")

    cum = np.asarray(cum)

    # Energy axis
    if x_keV is None:
        L = len(cum)
        x_keV = (start_eV + np.arange(L) * eV_per_ch) / 1000.0
    else:
        x_keV = np.asarray(x_keV)

    # Library
    if custom_lines_df is not None:
        lines_df = custom_lines_df.copy()
        # normalise energy column + label
        if "energy_keV" not in lines_df.columns and "E_keV" in lines_df.columns:
            lines_df = lines_df.rename(columns={"E_keV": "energy_keV"})
        if "E_keV" not in lines_df.columns and "energy_keV" in lines_df.columns:
            lines_df["E_keV"] = lines_df["energy_keV"]
        if "label" not in lines_df.columns:
            lines_df["label"] = lines_df["element"].astype(str) + " " + lines_df["line"].astype(str)
        lines_df = lines_df[lines_df["energy_keV"] <= beam_keV].sort_values("energy_keV", ignore_index=True)
    else:
        lines_df = line_library(beam_keV=beam_keV, elements=elements, include=include)

    # Peaks in keV space
    peaks_df, meta = detect_peaks(
        cum,
        x=x_keV,
        max_peaks=max_peaks,
        min_distance=min_distance
    )

    # Match
    fwhm_func = _fwhm_model_const(fwhm_Mn_eV)
    assign_df = _match_peaks_to_lines(peaks_df, lines_df, sigma=sigma, fwhm_func=fwhm_func)

    return assign_df, peaks_df, lines_df, meta


def infer_dataset_base_from_api(api_url: str) -> str:
    p = urlparse(api_url)
    qs = parse_qs(p.query)
    dataset = (qs.get("dataset") or [None])[0]
    if not dataset:
        raise ValueError("Could not find 'dataset' parameter in the API URL.")
    root = f"{p.scheme}://{p.netloc}/surface-viewer/data/"
    dataset_encoded = quote(dataset, safe="")
    return urljoin(root, dataset_encoded + "/")

def load_config_txt(dataset_base: str) -> dict:
    """
    Parse config.txt the same way the viewer does:
    - strip comments after '#'
    - split key=value
    - lowercase keys
    """
    url = urljoin(dataset_base, "config.txt")
    r = requests.get(url, timeout=30)
    if not r.ok:
        return {}
    txt = r.text
    cfg = {}
    for line in txt.splitlines():
        s = re.sub(r"#.*$", "", line).strip()
        if not s or "=" not in s:
            continue
        k, v = s.split("=", 1)
        cfg[k.strip().lower()] = v.strip()
    return cfg

def get_energy_cal_from_dataset(api_url: str,
                                default_eV_per_ch=20.000347,
                                default_start_eV=-192.768):
    dataset_base = infer_dataset_base_from_api(api_url)
    cfg = load_config_txt(dataset_base)

    eV_per_ch = float(cfg.get("eds_ev_per_ch", default_eV_per_ch))
    start_eV  = float(cfg.get("eds_start_ev", default_start_eV))

    n_channels = cfg.get("eds_n_channels", None)
    n_channels = int(n_channels) if n_channels and str(n_channels).isdigit() else None

    return {
        "dataset_base": dataset_base,
        "eV_per_ch": eV_per_ch,
        "start_eV": start_eV,
        "n_channels": n_channels,
        "raw_cfg": cfg
    }

def make_energy_axis(cum, cal: dict):
    n = len(cum) if cal.get("n_channels") is None else min(len(cum), cal["n_channels"])
    x_keV = (cal["start_eV"] + np.arange(n) * cal["eV_per_ch"]) / 1000.0
    return x_keV


def plot_identified_elements_confident(
    cum,
    api_url,
    assign_df,
    peaks_df=None,
    *,
    top_n_labels=8,
    beam_keV=15.0,
    fwhm_Mn_eV=67.8,
    sigma=2.5,
    show_all_markers=True,
    show_raw=True,
    show_corrected=True,
):
    """
    Confidence-first plot for aggregate EDS identification.
    - Uses dataset calibration from config.txt (viewer-compatible)
    - Shows raw + baseline-corrected aggregate
    - Marks detected peaks
    - Labels top N matches by a confidence score

    Parameters
    ----------
    cum : array-like
        Aggregate spectrum counts.
    api_url : str
        ROI API URL (used to infer dataset base + fetch config.txt).
    assign_df : pd.DataFrame
        Output from identify_elements (with lib_energy_keV, delta_keV, height/prominence where possible).
    peaks_df : pd.DataFrame, optional
        Output peaks table (from identify_elements call).
    top_n_labels : int
        How many labels to annotate.
    fwhm_Mn_eV : float
        Used for a constant resolution model in keV.
    sigma : float
        Matching tolerance used in the ID step; reused for scoring.
    """

    # --- Calibration from dataset config ---
    cal = get_energy_cal_from_dataset(api_url)
    eV_per_ch = cal["eV_per_ch"]
    start_eV  = cal["start_eV"]

    n = len(cum)
    if cal.get("n_channels"):
        n = min(n, cal["n_channels"])

    cum_plot = np.asarray(cum[:n], dtype=float)
    x_keV = (start_eV + np.arange(n) * eV_per_ch) / 1000.0

    # --- Preprocess for corrected display (same style as your peak pipeline) ---
    cum_corr = preprocess(cum_plot)

    # --- Guard ---
    if assign_df is None:
        assign_df = assign_df  # will fall through to empty branch

    # --- Confidence scoring ---
    # Convert FWHM -> sigma_E (keV)
    fwhm_keV = fwhm_Mn_eV / 1000.0
    sigma_E = fwhm_keV / 2.355 if fwhm_keV > 0 else 0.05

    assign = assign_df.copy() if hasattr(assign_df, "copy") else None

    if assign is not None and not assign.empty:
        # Ensure numeric columns exist
        if "height" not in assign.columns or assign["height"].isna().all():
            assign["height"] = assign.get("area", 0.0)

        if "prominence" not in assign.columns or assign["prominence"].isna().all():
            assign["prominence"] = assign["height"]

        if "delta_keV" not in assign.columns:
            # fallback if some earlier version didn't include it
            assign["delta_keV"] = assign["lib_energy_keV"] - assign["energy_keV"]

        # mismatch in units of detector sigma (smaller is better)
        assign["z_mismatch"] = (assign["delta_keV"].abs() / max(sigma_E, 1e-6))

        # Confidence score:
        # strong prominence boosted, penalised by mismatch
        assign["score"] = assign["prominence"] / (1.0 + assign["z_mismatch"])

        # Keep only lines in visible energy window
        assign = assign[(assign["lib_energy_keV"] >= x_keV[0]) & (assign["lib_energy_keV"] <= x_keV[-1])]

        # Pick top N by score
        label_df = (
            assign
            .sort_values("score", ascending=False)
            .head(int(top_n_labels))
            .sort_values("lib_energy_keV")
            .reset_index(drop=True)
        )
    else:
        label_df = None

    # --- Plot ---
    plt.figure(figsize=(10, 4.8))

    if show_raw:
        plt.plot(x_keV, cum_plot, lw=1, alpha=0.35, label="aggregate (raw)")

    if show_corrected:
        plt.plot(x_keV, cum_corr, lw=1.2, label="aggregate (baseline-corrected)")

    ymax = float(np.max(cum_corr if show_corrected else cum_plot)) if cum_plot.size else 1.0

    # Mark detected peak positions if provided
    if peaks_df is not None and not peaks_df.empty:
        # heights in peaks_df are on corrected curve in your pipeline
        try:
            y_peaks = cum_corr[peaks_df["idx"].to_numpy(dtype=int)]
        except Exception:
            y_peaks = None

        if y_peaks is not None:
            plt.scatter(peaks_df["x"], y_peaks, marker="x", zorder=5, label="detected peaks")

    # Draw all matched markers lightly
    if show_all_markers and assign is not None and not assign.empty:
        for _, r in assign.iterrows():
            plt.axvline(float(r["lib_energy_keV"]), ls="--", lw=0.6, alpha=0.18)

    # Emphasise + label top N confident matches
    if label_df is not None and not label_df.empty:
        # stagger text heights to reduce label collisions
        levels = np.linspace(0.92, 0.65, num=len(label_df))

        for i, (_, r) in enumerate(label_df.iterrows()):
            e = float(r["lib_energy_keV"])
            lbl = str(r.get("label", f"{r.get('element','')} {r.get('line','')}")).strip()

            plt.axvline(e, ls="--", lw=1.0, alpha=0.8)

            plt.text(
                e,
                levels[i] * ymax,
                lbl,
                rotation=90,
                va="top",
                ha="center",
                fontsize=9
            )

    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.title(f"Aggregate spectrum with top {top_n_labels} most confident labels")
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.show()

    # Return useful objects for debugging/display
    return {
        "cal": cal,
        "x_keV": x_keV,
        "cum_plot": cum_plot,
        "cum_corr": cum_corr,
        "assign_scored": assign if assign is not None else None,
        "label_df": label_df
    }


```

<!-- #region id="fba21ce6" -->
## 1. Load ROI selections
<!-- #endregion -->

```python id="b28f150b"

# Load ROI selections from the API
df = load_roi_api(api_url)
df.head()

```

```python id="6faccc2c"

if df.empty:
    print("No selections returned from the API.")
else:
    print(f"Total selected areas (ROI pixels): {len(df)}")
    if "basename" in df.columns:
        print(f"Unique basenames (sites/maps): {df['basename'].nunique()}")
    if "srcJson" in df.columns:
        print(f"Unique spectrum JSON files referenced: {df['srcJson'].nunique()}")

    # Simple per-site breakdown (if basename exists)
    if "basename" in df.columns:
        display(df.groupby("basename", dropna=False).size().sort_values(ascending=False).to_frame("n_selected").head(20))

```

<!-- #region id="4d6969eb" -->
## 2. Resolve spectrum URLs
<!-- #endregion -->

```python id="51396c39"

# Add fully-qualified JSON URLs
df = add_json_urls(df, api_url)
df[["row", "col", "basename", "foldername", "srcJson", "json_url"]].head()

```

<!-- #region id="efe3a060" -->
## 3. Download, attach, and aggregate spectra
<!-- #endregion -->

```python id="901c0168"

# Build spectrum index from the unique JSON files referenced in the ROI list
urls = df["json_url"].dropna().unique().tolist()
print(f"Will download {len(urls)} JSON file(s)")

index = build_spectrum_index(urls, progress=True)
df = attach_spectra(df, index, progress=True)

# How many selections successfully matched a spectrum?
ok = df["spectrum"].notna().sum()
print(f"Spectra attached for {ok} / {len(df)} selections")

# Stack + plot
stack, x = stack_spectra(df["spectrum"])
cum = plot_cumulative(stack, x, title_prefix="Cumulative ROI spectrum")
plot_overlay(stack, x, title="Overlay of ROI spectra")

```

<!-- #region id="97570b3b" -->
## 4. Simplest peak analysis on the aggregate
<!-- #endregion -->

```python id="T_iCmUgXKSew"
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Ensure "cum" exists from your aggregation cell
# and api_url is defined

# Get calibration (viewer-compatible)
cal = get_energy_cal_from_dataset(api_url)
x_keV = make_energy_axis(cum, cal)

# Align lengths
n = min(len(cum), len(x_keV))
cum_plot = np.asarray(cum[:n], dtype=float)
x_keV = np.asarray(x_keV[:n], dtype=float)

# Smooth then estimate baseline
cum_s = savgol_filter(cum_plot, 11, 3)
baseline = baseline_als(cum_s)

# Baseline-corrected
cum_corr = np.clip(cum_s - baseline, 0, None)

plt.figure(figsize=(10, 4.5))
plt.plot(x_keV, cum_plot, lw=1, alpha=0.35, label="raw aggregate")
plt.plot(x_keV, baseline, lw=1, label="estimated baseline (on smoothed)")
plt.plot(x_keV, cum_corr, lw=1.2, label="baseline-corrected")
plt.xlabel("Energy (keV)")
plt.ylabel("Counts")
plt.title("Baseline correction helps peak finding")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

```

```python id="300de1c6"

# Simple peak detection on the cumulative spectrum
peaks_df, meta = detect_peaks(cum, x=x, max_peaks=20)

print("Noise estimate:", meta["noise"])
print("min_prom:", meta["min_prom"])
print("min_height:", meta["min_height"])

peaks_df

```

```python id="d3c35dc9"

# Plot with detected peaks marked (using the baseline-corrected heights)
# We re-run preprocess so the marker heights are on the corrected curve.
cum_corr = preprocess(cum)
peaks_df_corr, _ = detect_peaks(cum, x=x, max_peaks=20)

plt.figure(figsize=(10, 4.5))
plt.plot(x, cum_corr, label="cumulative (corrected)")
if not peaks_df_corr.empty:
    plt.scatter(peaks_df_corr["x"], cum_corr[peaks_df_corr["idx"]], marker="x", label="peaks")
plt.xlabel("Channel")
plt.ylabel("Counts (baseline-corrected)")
plt.title("Aggregate spectrum – peak finding")
plt.tight_layout()
plt.show()

```

```python id="RX03sW0lBwEW"
# 1) calibration + axis
cal = get_energy_cal_from_dataset(api_url)
x_keV = make_energy_axis(cum, cal)

# 2) identify
assign_df, peaks_df, lines_df, meta = identify_elements(
    cum,
    x_keV=x_keV,
    beam_keV=15.0,
    fwhm_Mn_eV=67.8,
    max_peaks=25,
)

# 3) best confidence plot (easy knob)
top_n_labels = 8

_ = plot_identified_elements_confident(
    cum,
    api_url,
    assign_df,
    peaks_df=peaks_df,
    top_n_labels=top_n_labels,
    fwhm_Mn_eV=67.8,
    show_all_markers=True,
);

```

```python id="BJ08NVUOJl76"

```
