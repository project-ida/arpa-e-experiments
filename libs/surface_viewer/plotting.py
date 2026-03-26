from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .calibration import channel_to_keV, get_energy_cal_from_dataset, make_energy_axis_from_length
from .peaks import preprocess


def get_plot_axis(n, use_keV=False, cal=None):
    if use_keV:
        if cal is None:
            raise ValueError("Calibration dictionary is required when use_keV=True")
        return make_energy_axis_from_length(n, cal), "Energy (keV)"
    return np.arange(n), "Channel"


def get_band_span(band_start, band_end, use_keV=False, cal=None):
    if use_keV:
        if cal is None:
            raise ValueError("Calibration dictionary is required when use_keV=True")
        lo = channel_to_keV(band_start, cal)
        hi = channel_to_keV(band_end, cal)
        return min(lo, hi), max(lo, hi)
    return min(band_start, band_end), max(band_start, band_end)


def add_energy_top_axis(ax, cal: dict, n=None):
    if cal is None:
        return None
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    ticks = np.asarray(ax.get_xticks(), dtype=float)
    if n is not None:
        ticks = ticks[(ticks >= 0) & (ticks <= n - 1)]
    top.set_xticks(ticks)
    top.set_xticklabels([f"{channel_to_keV(t, cal):.2f}" for t in ticks])
    top.set_xlabel("Energy (keV)")
    return top


def plot_cumulative(stack, x, *, title_prefix="Cumulative spectrum", xlabel="Channel"):
    cum = stack.sum(axis=0)
    n = stack.shape[0]
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, cum)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(f"{title_prefix} (n={n})")
    plt.tight_layout()
    plt.show()
    return cum


def plot_overlay(
    stack,
    x,
    *,
    max_traces=50,
    title="Overlay spectra",
    xlabel="Channel",
    cal=None,
    show_energy_top_axis=False,
):
    n = stack.shape[0]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    step = max(1, n // max_traces)
    for i in range(0, n, step):
        ax.plot(x, stack[i], alpha=0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(f"{title} (showing ~{min(n, max_traces)} of {n})")
    if show_energy_top_axis and cal is not None and str(xlabel).lower().startswith("channel"):
        add_energy_top_axis(ax, cal=cal, n=len(x))
    fig.tight_layout()
    plt.show()


def plot_with_peaks(y, x=None, peaks_df=None, title="Spectrum with detected peaks", xlabel="Channel"):
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
    """Confidence-first plot for aggregate EDS identification."""
    cal = get_energy_cal_from_dataset(api_url)
    eV_per_ch = cal["eV_per_ch"]
    start_eV = cal["start_eV"]

    n = len(cum)
    if cal.get("n_channels"):
        n = min(n, cal["n_channels"])

    cum_plot = np.asarray(cum[:n], dtype=float)
    x_keV = (start_eV + np.arange(n) * eV_per_ch) / 1000.0

    cum_corr = preprocess(cum_plot)

    assign = assign_df.copy() if hasattr(assign_df, "copy") else None

    fwhm_keV = fwhm_Mn_eV / 1000.0
    sigma_E = fwhm_keV / 2.355 if fwhm_keV > 0 else 0.05

    if assign is not None and not assign.empty:
        if "height" not in assign.columns or assign["height"].isna().all():
            assign["height"] = assign.get("area", 0.0)

        if "prominence" not in assign.columns or assign["prominence"].isna().all():
            assign["prominence"] = assign["height"]

        if "delta_keV" not in assign.columns:
            assign["delta_keV"] = assign["lib_energy_keV"] - assign["energy_keV"]

        assign["z_mismatch"] = assign["delta_keV"].abs() / max(sigma_E, 1e-6)
        assign["score"] = assign["prominence"] / (1.0 + assign["z_mismatch"])

        assign = assign[(assign["lib_energy_keV"] >= x_keV[0]) & (assign["lib_energy_keV"] <= x_keV[-1])]

        label_df = (
            assign.sort_values("score", ascending=False)
            .head(int(top_n_labels))
            .sort_values("lib_energy_keV")
            .reset_index(drop=True)
        )
    else:
        label_df = None

    plt.figure(figsize=(10, 4.8))

    if show_raw:
        plt.plot(x_keV, cum_plot, lw=1, alpha=0.35, label="aggregate (raw)")

    if show_corrected:
        plt.plot(x_keV, cum_corr, lw=1.2, label="aggregate (baseline-corrected)")

    ymax = float(np.max(cum_corr if show_corrected else cum_plot)) if cum_plot.size else 1.0

    if peaks_df is not None and not peaks_df.empty:
        try:
            y_peaks = cum_corr[peaks_df["idx"].to_numpy(dtype=int)]
        except Exception:
            y_peaks = None

        if y_peaks is not None:
            plt.scatter(peaks_df["x"], y_peaks, marker="x", zorder=5, label="detected peaks")

    if show_all_markers and assign is not None and not assign.empty:
        for _, r in assign.iterrows():
            plt.axvline(float(r["lib_energy_keV"]), ls="--", lw=0.6, alpha=0.18)

    if label_df is not None and not label_df.empty:
        levels = np.linspace(0.92, 0.65, num=len(label_df))

        for i, (_, r) in enumerate(label_df.iterrows()):
            e = float(r["lib_energy_keV"])
            lbl = str(r.get("label", f"{r.get('element', '')} {r.get('line', '')}")).strip()

            plt.axvline(e, ls="--", lw=1.0, alpha=0.8)
            plt.text(e, levels[i] * ymax, lbl, rotation=90, va="top", ha="center", fontsize=9)

    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.title(f"Aggregate spectrum with top {top_n_labels} most confident labels")
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.show()

    return {
        "cal": cal,
        "x_keV": x_keV,
        "cum_plot": cum_plot,
        "cum_corr": cum_corr,
        "assign_scored": assign if assign is not None else None,
        "label_df": label_df,
    }
