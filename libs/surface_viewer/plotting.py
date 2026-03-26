from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _select_overlay_indices(n_cells: int, n_show: int) -> np.ndarray:
    if n_cells <= 0:
        return np.array([], dtype=int)
    n_show = min(max(int(n_show), 1), n_cells)
    if n_cells <= n_show:
        return np.arange(n_cells, dtype=int)
    return np.unique(np.linspace(0, n_cells - 1, n_show, dtype=int))


def plot_overlaid_cell_spectra(
    aggregate_results,
    peak_summary,
    band_start,
    band_end,
    band_mode="channels",
    cal=None,
    show_energy_top_axis=True,
    n_overlay_spectra=50,
    peak_top_n_labels=6,
    peak_fwhm_mn_ev=67.8,
    show_peak_crosses=False,
    show_element_lines=True,
    band_color="gray",
    band_alpha=0.15,
    line_color="black",
    overlay_color="C0",
):
    """Overlay up to N cell spectra for each group with optional band shading and element-line annotations."""
    from .spectra import band_label_text, resolve_band_to_channels

    band_start_ch, band_end_ch = resolve_band_to_channels(
        band_start,
        band_end,
        band_mode=band_mode,
        cal=cal,
    )
    band_label = band_label_text(band_start, band_end, band_mode)

    for name, res in aggregate_results.items():
        stack = np.asarray(res["stack"], dtype=float)
        if stack.ndim != 2 or stack.shape[0] == 0:
            print(f"Skipping {name}: no spectra available.")
            continue

        show_idx = _select_overlay_indices(stack.shape[0], n_overlay_spectra)
        x = np.arange(stack.shape[1], dtype=float)

        fig, ax = plt.subplots(figsize=(11, 5))
        for idx in show_idx:
            ax.plot(x, stack[idx], lw=0.8, alpha=0.25, color=overlay_color)

        ax.axvspan(band_start_ch, band_end_ch, color=band_color, alpha=band_alpha)

        ymax = float(np.nanmax(stack[show_idx])) if show_idx.size else 1.0
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0

        if show_peak_crosses:
            peaks_df_overlay = peak_summary.get(name, {}).get("peaks_df", pd.DataFrame())
            if peaks_df_overlay is not None and not peaks_df_overlay.empty and "x" in peaks_df_overlay.columns:
                x_peaks = peaks_df_overlay["x"].to_numpy(dtype=float)
                x_peaks = x_peaks[(x_peaks >= 0) & (x_peaks <= stack.shape[1] - 1)]
                if x_peaks.size:
                    ax.scatter(
                        x_peaks,
                        np.full_like(x_peaks, 0.98 * ymax),
                        marker="x",
                        s=28,
                        color=line_color,
                        zorder=5,
                    )

        assign_df_overlay = peak_summary.get(name, {}).get("assign_df", pd.DataFrame())
        if show_element_lines and cal is not None and assign_df_overlay is not None and not assign_df_overlay.empty:
            assign = assign_df_overlay.copy()

            if "height" not in assign.columns or assign["height"].isna().all():
                assign["height"] = assign.get("area", 0.0)

            if "prominence" not in assign.columns or assign["prominence"].isna().all():
                assign["prominence"] = assign["height"]

            if "delta_keV" not in assign.columns and {"lib_energy_keV", "energy_keV"}.issubset(assign.columns):
                assign["delta_keV"] = assign["lib_energy_keV"] - assign["energy_keV"]

            if "score" not in assign.columns:
                if "delta_keV" in assign.columns:
                    sigma_E = max((float(peak_fwhm_mn_ev) / 1000.0) / 2.355, 1e-6)
                    assign["z_mismatch"] = assign["delta_keV"].abs() / sigma_E
                else:
                    assign["z_mismatch"] = 0.0
                assign["score"] = assign["prominence"] / (1.0 + assign["z_mismatch"])

            assign = assign.sort_values("score", ascending=False).head(int(peak_top_n_labels))
            assign = assign.sort_values("lib_energy_keV").reset_index(drop=True)

            levels = np.linspace(0.93, 0.68, num=max(len(assign), 1))
            start_eV = float(cal["start_eV"])
            eV_per_ch = float(cal["eV_per_ch"])

            for i, (_, row) in enumerate(assign.iterrows()):
                e_keV = float(row["lib_energy_keV"])
                ch = (e_keV * 1000.0 - start_eV) / eV_per_ch
                if 0 <= ch <= stack.shape[1] - 1:
                    label = str(row.get("label", f"{row.get('element', '')} {row.get('line', '')}")).strip()
                    ax.axvline(ch, ls="--", lw=1.0, alpha=0.55, color=line_color)
                    ax.text(
                        ch,
                        levels[min(i, len(levels) - 1)] * ymax,
                        label,
                        rotation=90,
                        va="top",
                        ha="center",
                        fontsize=8,
                        color=line_color,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.6),
                    )

        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts")
        ax.set_title(f"{name}: {len(show_idx)} overlaid cell spectra ({band_label}; shaded gray)")

        if show_energy_top_axis and cal is not None:
            add_energy_top_axis(ax, cal=cal, n=stack.shape[1])

        fig.tight_layout()
        plt.show()


def plot_stacked_band_histograms(
    aggregate_results,
    shared_bins,
    *,
    band_label,
    global_min=None,
    global_max=None,
    figsize_per_row=3.2,
):
    """Plot one band-sum histogram per group in a vertical stack with a shared x-axis."""
    n = len(aggregate_results)
    fig, axes = plt.subplots(n, 1, figsize=(10, figsize_per_row * n), sharex=True, squeeze=False)
    axes = axes[:, 0]

    if global_min is None or global_max is None:
        all_band_vals = np.concatenate([
            res["df"]["band_value"].dropna().to_numpy(dtype=float)
            for res in aggregate_results.values()
            if len(res["df"]) > 0
        ])
        global_min = float(np.min(all_band_vals))
        global_max = float(np.max(all_band_vals))

    for i, (ax, (name, res)) in enumerate(zip(axes, aggregate_results.items())):
        vals = res["df"]["band_value"].dropna().to_numpy(dtype=float)
        ax.hist(vals, bins=shared_bins)
        ax.set_xlim(global_min, global_max)
        ax.set_ylabel("Number of cells")
        ax.set_title(f"{name}\n{band_label}")
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xlabel("Raw band sum")
    fig.tight_layout()
    plt.show()


def plot_overlay_band_histograms(
    aggregate_results,
    shared_bins,
    *,
    band_label,
    vmin=None,
    vmax=None,
    log_y=True,
    shade_selected_range=True,
    legend_loc="upper left",
):
    """Overlay band-sum histograms across groups, optionally shading a selected value range."""
    plt.figure(figsize=(10, 5))

    for name, res in aggregate_results.items():
        vals = res["df"]["band_value"].dropna().to_numpy(dtype=float)
        plt.hist(vals, bins=shared_bins, alpha=0.40, label=name)

    if shade_selected_range and vmin is not None and vmax is not None:
        plt.axvspan(vmin, vmax, color="gray", alpha=0.15, label=f"selected range: {vmin}–{vmax}")

    if log_y:
        plt.yscale("log")

    plt.xlabel("Raw band sum")
    plt.ylabel("Number of cells")
    plt.title(f"Band-sum comparison, {band_label}")
    plt.legend(frameon=True, facecolor="white", framealpha=1.0, edgecolor="lightgray", loc=legend_loc)
    plt.tight_layout()
    plt.show()
