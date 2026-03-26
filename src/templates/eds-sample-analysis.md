---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/templates/eds-sample-analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/templates/eds-sample-analysis.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>

<!-- #region id="7f188c81" -->
*Overview: This cell explains the notebook purpose and basic usage.*

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

```python colab={"base_uri": "https://localhost:8080/"} id="_K48SSZ_Z44M" outputId="bbd9ec0a-8fc3-43e0-cd58-d73d8376ca3c"
# Set up the GitHub repo in Colab so the shared helper modules can be imported.

!pip install colocal -q
import colocal
root, branch, cwd = colocal.setup("https://github.com/project-ida/arpa-e-experiments")
```

```python id="gXbQnjaTaBrB"
# Import Python libraries, URL helpers, and reusable Surface Viewer analysis modules.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlparse, parse_qs, quote_plus

from libs.surface_viewer.io import (
    get_roi_name_from_api_url,
    load_roi_api,
    add_json_urls,
    load_all_cells_from_selection_grid,
    build_spectrum_index,
    attach_spectra,
)

from libs.surface_viewer.spectra import (
    stack_spectra_trim,
    band_sum,
    summarize_band_values,
    resolve_band_to_channels,
    band_label_text,
)

from libs.surface_viewer.calibration import (
    maybe_get_calibration,
    channel_to_keV,
    make_energy_axis,
)

from libs.surface_viewer.peaks import (
    baseline_als,
    preprocess,
    detect_peaks,
    identify_elements,
)

from libs.surface_viewer.plotting import (
    add_energy_top_axis,
    plot_cumulative,
    plot_overlay,
    plot_with_peaks,
    plot_identified_elements_confident,
    plot_overlaid_cell_spectra,
)

from libs.surface_viewer.overlays import (
    get_api_auth,
    create_overlay,
    delete_overlay,
)
```

```python id="77a85a02"
# Define ROI URLs and the main analysis parameters for loading, calibration, and peak finding.

ROI_API_URLS = [
    "https://nucleonics.mit.edu/surface-viewer/api/rois.php?dataset=JPB1_Pd-TF-11_EDS_post%20(20250825_sample%209%20(eds))&name=sample-corner",
    "https://nucleonics.mit.edu/surface-viewer/api/rois.php?dataset=JPB1_Pd-TF-11_EDS_post%20(20250825_sample%209%20(eds))&name=margin",
]

# General display / loading behavior
LOAD_ALL_CELLS = True
SHOW_ENERGY_TOP_AXIS = True
ALLOW_DEFAULT_CALIBRATION = True

# Peak-analysis knobs
PEAK_MAX_PEAKS = 6
PEAK_TOP_N_LABELS = 6
PEAK_BEAM_KEV = 15.0
PEAK_FWHM_MN_EV = 67.8

```

<!-- #region id="6b0ef3b1" -->
## 1. Load ROI selections and, optionally, the full mosaic cell grid

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 635} id="9df52624" outputId="13f6eba4-2cb7-4b69-9c74-6a9dbde517b9"
# Load ROI selections and, if requested, enumerate all cells from overlays/selection-grid.json.

roi_frames = {}
all_cells_df = pd.DataFrame()

for api_url in ROI_API_URLS:
    roi_name = get_roi_name_from_api_url(api_url)
    df = load_roi_api(api_url)
    df = add_json_urls(df, api_url)
    df["roi_name"] = roi_name
    roi_frames[roi_name] = df

    print(f"{roi_name}: {len(df)} selected cells")
    if not df.empty:
        display(df.head())

if LOAD_ALL_CELLS:
    all_cells_df = load_all_cells_from_selection_grid(ROI_API_URLS[0])
    print(f"all-cells: {len(all_cells_df)} cells enumerated from overlays/selection-grid.json")
    if not all_cells_df.empty:
        display(all_cells_df.head())

```

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["dc05a57d2a2a4baa8c7ce26a0c9ba5b1", "d9a69d6e5d114f84ba97f446072130ef", "252d540c9aa541f6a6bbe33649e8f75a", "799625fd8cdc44709e540b233cc4ee21", "bfc76807faf5494b89edaa18c3460a65", "de0c0083034947608b493fe6e486c7c7", "3f36a5b55bca4338be4a84bdd482bce5", "5a3adf09ae0c469b9ac86a0321b3da2a", "063881938ff34fecaf164b79f4b463d4", "f077cfd657a8406c9dcd0c972c961605", "c661b1cef3a1462796195bebf95d440d"]} id="f7c5adb5" outputId="db6c5080-2eee-4fac-f863-0ffa04f3e712"
# Collect the unique spectrum JSON URLs needed across ROI selections and the all-cells grid.

roi_urls = sorted(set(
    url
    for df in roi_frames.values()
    for url in df["json_url"].dropna().unique().tolist()
))

all_cell_urls = []
if LOAD_ALL_CELLS and not all_cells_df.empty:
    all_cell_urls = sorted(all_cells_df["json_url"].dropna().unique().tolist())

all_urls = sorted(set(roi_urls) | set(all_cell_urls))

print(f"Will download {len(all_urls)} unique JSON file(s) across ROI sets and the all-cells grid")
index = build_spectrum_index(all_urls, progress=True)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 165, "referenced_widgets": ["6856240dba6141c7b8c299a9691dc031", "5cc6d421fd7549608cf7d749d441d826", "2ba857add00e414d862c107307fa5541", "4f890fa889c448e4b483113761b9df49", "3f16af0aebe84135a9af6c0ed3fc23c0", "dca688518b7446d09e0d7a8fdcccfdb5", "70d0856771cb4ffa939b8432c06477d4", "14d968100d5f488ab94bb177160a7acc", "87aa801639974efcb8e6a679f2c8b6f3", "e3e8ae03ccc34dbe83224f785785523b", "475bd3d679b24efc985843408a446265", "947fd897bb174ee2beb655a8fefa519b", "1841091ce54b47208f0ccb820ee5141e", "5c3a597045ad4837b519f03f6db772e7", "8f1de4d8b4234b54a2fd8f5f4ca204e1", "ff9daa35f2a24e5b8d2602d2593a60db", "398f6d69b14c49558c71102055238bfe", "0eb08f0c3bb94ce4afbb09b99180e3c8", "81cef18e40324682b795fdd29e067b37", "ee707a064fbd428598493a10e9e14974", "71f58774f3f64f279fc5ace2a9552f93", "c85a1aa269424f1ea7afd871bd0a6b81", "45bd0cc8b8934585b169b7b503602acb", "19d021851e6b4aacb9b994ff0c6b36ca", "fb701fa531e24368878a8b95a12543bc", "14d7660602254bedb2355af4e0aa65db", "b12cc30d34cc495aba2fc90be82cc4fe", "b1d8efd5ac3444b6845a14f2060ebbfc", "be2378d133224032b2697a68e3088ac8", "eba9a547d43d46a5b533d69da5ce66f7", "7ea11d36344c4ee09ada76a63883f44f", "d6010fb800164f1aa1dbe93f6b0b430b", "c0055c152f0f407db81acf44bb5a1169"]} id="83346e85" outputId="f09c9ff8-112f-462b-88e4-449fd819495d"
# Attach spectra to the ROI tables and the optional all-cells table.

for roi_name, df in roi_frames.items():
    roi_frames[roi_name] = attach_spectra(df, index, progress=True)

    ok = roi_frames[roi_name]["spectrum"].notna().sum()
    print(f"{roi_name}: spectra attached for {ok} / {len(roi_frames[roi_name])} selections")

if LOAD_ALL_CELLS and not all_cells_df.empty:
    all_cells_df = attach_spectra(all_cells_df, index, progress=True)
    ok_all = all_cells_df["spectrum"].notna().sum()
    print(f"all-cells: spectra attached for {ok_all} / {len(all_cells_df)} cells")

```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="6653d088" outputId="f683a84b-13e5-49ab-f664-49b0f8fdb6fd"
# Print quick summaries of the selected ROI cells by basename and source spectrum file.

for roi_name, df in roi_frames.items():
    print(f"\n=== {roi_name} ===")
    if df.empty:
        print("No selections")
        continue

    print(f"Selected cells: {len(df)}")
    if "basename" in df.columns:
        print(f"Unique basenames: {df['basename'].nunique()}")
    if "srcJson" in df.columns:
        print(f"Unique spectrum JSON files: {df['srcJson'].nunique()}")

    display(df.groupby("basename", dropna=False).size().sort_values(ascending=False).to_frame("n_selected").head(20))

if LOAD_ALL_CELLS and not all_cells_df.empty:
    print("\n=== all-cells ===")
    print(f"Cells in full mosaic: {len(all_cells_df)}")
    if "basename" in all_cells_df.columns:
        print(f"Unique basenames: {all_cells_df['basename'].nunique()}")
    if "srcJson" in all_cells_df.columns:
        print(f"Unique spectrum JSON files: {all_cells_df['srcJson'].nunique()}")

```

<!-- #region id="8ed9b876" -->
*Step 2: Build aggregate spectra for all cells and each ROI region.*

## 2. Build aggregate spectra for all cells and for each ROI region

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="b97f6ba8" outputId="b9c111ac-d858-40a8-b3e1-a97a6f96fea7"
# Build aggregate stacks, cumulative spectra, and mean spectra for each analysis group.

need_calibration = True
cal = maybe_get_calibration(
    ROI_API_URLS,
    need_calibration=need_calibration,
    allow_defaults=ALLOW_DEFAULT_CALIBRATION,
)

roi_results = {}
for api_url in ROI_API_URLS:
    roi_name = get_roi_name_from_api_url(api_url)
    df = roi_frames[roi_name]
    df_ok = df[df["spectrum"].notna()].copy()

    if df_ok.empty:
        print(f"{roi_name}: no spectra available")
        continue

    stack, x = stack_spectra_trim(df_ok["spectrum"])
    roi_results[roi_name] = {
        "label": roi_name,
        "api_url": api_url,
        "df": df_ok,
        "stack": stack,
        "x": x,
        "cumulative": stack.sum(axis=0),
        "mean_spectrum": stack.mean(axis=0),
    }
    print(f"{roi_name}: {stack.shape[0]} spectra, common length {stack.shape[1]}")

all_cells_result = None
if LOAD_ALL_CELLS and not all_cells_df.empty:
    df_all_ok = all_cells_df[all_cells_df["spectrum"].notna()].copy()
    if not df_all_ok.empty:
        stack_all, x_all = stack_spectra_trim(df_all_ok["spectrum"])
        all_cells_result = {
            "label": "all-cells",
            "api_url": ROI_API_URLS[0],
            "df": df_all_ok,
            "stack": stack_all,
            "x": x_all,
            "cumulative": stack_all.sum(axis=0),
            "mean_spectrum": stack_all.mean(axis=0),
        }
        print(f"all-cells: {stack_all.shape[0]} spectra, common length {stack_all.shape[1]}")

aggregate_results = {}
if all_cells_result is not None:
    aggregate_results["all-cells"] = all_cells_result
for api_url in ROI_API_URLS:
    roi_name = get_roi_name_from_api_url(api_url)
    if roi_name in roi_results:
        aggregate_results[roi_name] = roi_results[roi_name]

```

```python colab={"base_uri": "https://localhost:8080/"} id="0ef9187d" outputId="2ec4d208-754c-4a81-d302-e86cffc61b40"
# Report which energy calibration was found and whether config values or defaults are being used.

if cal is not None:
    source_text = "config.txt" if cal.get("from_config") else "defaults"
    print(f"Energy calibration source: {source_text}")
    print("Calibration:", cal)
else:
    print("No energy calibration available; top energy axis will be omitted.")

```

```python colab={"base_uri": "https://localhost:8080/", "height": 506} id="b2b49706" outputId="d4626af1-97b9-4f46-fea9-2aa4636d580d"
# Plot cumulative spectra for all cells and ROI regions on a common axis.

fig, ax = plt.subplots(figsize=(11, 5))

for name, res in aggregate_results.items():
    y = res["cumulative"]
    x_plot = np.arange(len(y))
    lw = 2.2 if name == "all-cells" else 1.5
    ax.plot(x_plot, y, label=f"{name} total (n={res['stack'].shape[0]})", lw=lw)

ax.set_xlabel("Channel")
ax.set_ylabel("Counts")
ax.set_title("Cumulative spectra: all cells and selected ROI regions")

if SHOW_ENERGY_TOP_AXIS and cal is not None:
    max_len = max(len(res["cumulative"]) for res in aggregate_results.values())
    add_energy_top_axis(ax, cal=cal, n=max_len)

ax.legend(frameon=False)
fig.tight_layout()
plt.show()

```

<!-- #region id="00ef80b8" -->
*Step 2b: Run basic conditioning and first-pass element identification on the aggregate spectra.*

### Basic conditioning and peak identification on the aggregate spectra

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="c9ae05a3" outputId="a1d4eb82-bc34-4eb1-d9ca-7bd2ecdd33cb"
# Condition each aggregate spectrum, detect peaks, and print first-pass element assignments.

peak_summary = {}

for name, res in aggregate_results.items():
    print(f"\n=== {name}: cumulative spectrum ===")
    cum = np.asarray(res["cumulative"], dtype=float)
    x = np.arange(len(cum))

    # Plot raw cumulative spectrum with channel axis and optional energy top axis
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, cum, lw=1.2, label=f"{name} cumulative")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")
    ax.set_title(f"{name}: cumulative spectrum")
    if SHOW_ENERGY_TOP_AXIS and cal is not None:
        add_energy_top_axis(ax, cal=cal, n=len(cum))
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

    # Baseline/corrected view (same spirit as the demo notebook)
    baseline = baseline_als(cum)
    corrected = np.clip(cum - baseline, 0, None)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, cum, lw=1.0, alpha=0.4, label="raw aggregate")
    ax.plot(x, baseline, lw=1.0, label="estimated baseline")
    ax.plot(x, corrected, lw=1.2, label="baseline-corrected")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")
    ax.set_title(f"{name}: baseline correction")
    if SHOW_ENERGY_TOP_AXIS and cal is not None:
        add_energy_top_axis(ax, cal=cal, n=len(cum))
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

    # Peak detection table
    peaks_df, meta = detect_peaks(cum, x=x, max_peaks=PEAK_MAX_PEAKS)
    print("Noise estimate:", meta["noise"])
    print("min_prom:", meta["min_prom"])
    print("min_height:", meta["min_height"])
    display(peaks_df)

    # Plot corrected spectrum with peaks marked
    # fig, ax = plt.subplots(figsize=(11, 4.5))
    # ax.plot(x, corrected, label="cumulative (corrected)")
    # if not peaks_df.empty:
    #     ax.scatter(peaks_df["x"], corrected[peaks_df["idx"]], marker="x", label="peaks")
    # ax.set_xlabel("Channel")
    # ax.set_ylabel("Counts (baseline-corrected)")
    # ax.set_title(f"{name}: peak finding")
    # if SHOW_ENERGY_TOP_AXIS and cal is not None:
    #     add_energy_top_axis(ax, cal=cal, n=len(cum))
    # ax.legend(frameon=False)
    # fig.tight_layout()
    # plt.show()

    assign_df = pd.DataFrame()
    lines_df = pd.DataFrame()
    if cal is not None:
        x_keV = make_energy_axis(cum, cal)
        assign_df, peaks_df_id, lines_df, meta_id = identify_elements(
            cum,
            x_keV=x_keV,
            beam_keV=PEAK_BEAM_KEV,
            fwhm_Mn_eV=PEAK_FWHM_MN_EV,
            max_peaks=PEAK_MAX_PEAKS,
        )
        display(assign_df.head(30))
        _ = plot_identified_elements_confident(
            cum,
            res["api_url"],
            assign_df,
            peaks_df=peaks_df_id,
            top_n_labels=PEAK_TOP_N_LABELS,
            fwhm_Mn_eV=PEAK_FWHM_MN_EV,
            show_all_markers=True,
        )
    else:
        print("Skipping element identification because no energy calibration is available.")

    peak_summary[name] = {
        "peaks_df": peaks_df,
        "assign_df": assign_df,
        "lines_df": lines_df,
        "meta": meta,
    }

```

<!-- #region id="ad9dc613" -->
*Step 3: Choose a spectral band and compare band-sum distributions.*

## 3. Choose a spectral band and then create band-sum histograms

<!-- #endregion -->

```python id="e770b768"
# Define the spectral band to use for band-sum histograms and overlays.

# Band definition for band-sum histograms / overlays.
# BAND_MODE can be "channels" or "keV".
BAND_MODE = "kev" #"channels"
BAND_START = 4.8 #500
BAND_END = 9.8 #1000

HIST_BINS = 500
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="WyWhFSHHjWzj" outputId="493cdfcb-6063-4924-e937-e70dc268c536"
# Overlay up to 50 individual cell spectra per region, with element annotations and the selected band shaded in gray.

N_OVERLAY_SPECTRA = 50

plot_overlaid_cell_spectra(
    aggregate_results=aggregate_results,
    peak_summary=peak_summary,
    band_start=BAND_START,
    band_end=BAND_END,
    band_mode=BAND_MODE,
    cal=cal,
    show_energy_top_axis=SHOW_ENERGY_TOP_AXIS,
    n_overlay_spectra=N_OVERLAY_SPECTRA,
    peak_top_n_labels=PEAK_TOP_N_LABELS,
    peak_fwhm_mn_ev=PEAK_FWHM_MN_EV,
    show_peak_crosses=False,
    show_element_lines=True,
    band_color="gray",
    band_alpha=0.15,
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="31a290de" outputId="50a081fa-44d4-4737-f875-861b7229e7c6"
# Resolve the requested band to channel indices and compute per-cell band sums.

band_start_ch, band_end_ch = resolve_band_to_channels(
    BAND_START,
    BAND_END,
    band_mode=BAND_MODE,
    cal=cal,
)

print(f"Resolved analysis band: {band_start_ch}–{band_end_ch} channels")
if str(BAND_MODE).lower() == "kev":
    print(f"Requested in energy units: {BAND_START}–{BAND_END} keV")

if cal is not None:
    band_lo_keV = channel_to_keV(band_start_ch, cal)
    band_hi_keV = channel_to_keV(band_end_ch, cal)
    source_text = "config.txt" if cal.get("from_config") else "defaults"
    print(f"Resolved energy span: {band_lo_keV:.3f}–{band_hi_keV:.3f} keV ({source_text})")

for name, res in aggregate_results.items():
    res["df"] = res["df"].copy()
    res["df"]["band_value"] = res["df"]["spectrum"].apply(lambda s: band_sum(s, band_start_ch, band_end_ch))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 967} id="12d03a2b" outputId="bee09a78-2d62-4c6f-e4df-9b1029b1e1ca"
# Build vertically stacked histograms with shared bins and a shared global x-axis range.

# Shared bins across all aggregates
all_band_vals = np.concatenate([
    res["df"]["band_value"].dropna().to_numpy(dtype=float)
    for res in aggregate_results.values()
    if len(res["df"]) > 0
])

global_min = float(np.min(all_band_vals))
global_max = float(np.max(all_band_vals))

if global_max > global_min:
    shared_bins = np.linspace(global_min, global_max, HIST_BINS + 1)
else:
    shared_bins = np.array([global_min - 0.5, global_max + 0.5])

n = len(aggregate_results)

fig, axes = plt.subplots(
    n, 1,
    figsize=(10, 3.2 * n),
    sharex=True,
    squeeze=False
)
axes = axes[:, 0]

for i, (ax, (name, res)) in enumerate(zip(axes, aggregate_results.items())):
    vals = res["df"]["band_value"].dropna().to_numpy(dtype=float)
    ax.hist(vals, bins=shared_bins)

    ax.set_xlim(global_min, global_max)
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{name}\n{band_label_text(BAND_START, BAND_END, BAND_MODE)}")

    if i < n - 1:
        ax.tick_params(axis="x", labelbottom=False)

axes[-1].set_xlabel("Raw band sum")

plt.tight_layout()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="c1c277c6" outputId="343fe425-4937-415c-d36c-4bffc9f1b548"
# Overlay all band-sum histograms on one log-scale plot for direct comparison.

plt.figure(figsize=(10, 5))

for name, res in aggregate_results.items():
    vals = res["df"]["band_value"].dropna().to_numpy(dtype=float)
    plt.hist(vals, bins=shared_bins, alpha=0.40, label=name)

plt.yscale("log")
plt.xlabel("Raw band sum")
plt.ylabel("Number of cells")
plt.title(f"Band-sum comparison, {band_label_text(BAND_START, BAND_END, BAND_MODE)}")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 332} id="a746c947" outputId="878b9bf6-5eea-48bd-97ec-1556b5f5f3f9"
# Summarize the band-sum distributions and propose value ranges for overlays.

stats_rows = []

for name, res in aggregate_results.items():
    stats = summarize_band_values(res["df"]["band_value"])
    stats["roi_name"] = name
    stats_rows.append(stats)

stats_df = pd.DataFrame(stats_rows).set_index("roi_name")

range_suggestions_df = pd.DataFrame({
    "global_min": int(round(global_min)),
    "global_max": int(round(global_max)),
    "vmin_mild": stats_df["p05"].round().astype(int),
    "vmax_mild": stats_df["p95"].round().astype(int),
    "vmin_strong": stats_df["p10"].round().astype(int),
    "vmax_strong": stats_df["p90"].round().astype(int),
    "threshold_candidate": stats_df["p10"].round().astype(int),
}, index=stats_df.index)

display(stats_df.round(3))
display(range_suggestions_df)
```

<!-- #region id="9836439c" -->
*Step 4: Optionally create or delete overlay heatmap files in the Surface Viewer.*

## 4. Optional: create overlay heatmap files

<!-- #endregion -->

```python id="mN5MxZpZF3A8"
# Choose the display range to apply when creating the overlay heatmap.

# Pick one of the suggested ranges from range_suggestions_df above, then edit if desired.
VMIN = 3
VMAX = 9039
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="5tVt5_NzI30V" outputId="d7b93941-ae7c-4e80-e63a-f12aec7e3ed1"
# Preview the chosen overlay range on the combined log-scale band-sum histogram.

plt.figure(figsize=(10, 5))

for name, res in aggregate_results.items():
    vals = res["df"]["band_value"].dropna().to_numpy(dtype=float)
    plt.hist(vals, bins=shared_bins, alpha=0.40, label=name)

plt.axvspan(VMIN, VMAX, color="gray", alpha=0.15, label=f"selected range: {VMIN}–{VMAX}")

plt.yscale("log")
plt.xlabel("Raw band sum")
plt.ylabel("Number of cells")
plt.title(f"Band-sum comparison, {band_label_text(BAND_START, BAND_END, BAND_MODE)}")
plt.legend(frameon=True, facecolor="white", framealpha=1.0, edgecolor="lightgray", loc="upper left")
plt.tight_layout()
plt.show()
```

```python id="33caf51e"
# Authenticate to the overlay API and extract dataset metadata from the ROI URL.

auth = get_api_auth()

dataset = parse_qs(urlparse(ROI_API_URLS[0]).query)["dataset"][0]
input_folder = "aggregated-spectra"

print("Dataset:", dataset)
print("Input folder:", input_folder)
print(f"Using resolved band: {band_start_ch}–{band_end_ch} channels")
```

```python id="599dffb5"
# Create the overlay heatmap and print both the overlay URL and the sample viewer link.

resp_create = create_overlay(
    auth=auth,
    dataset=dataset,
    input_folder=input_folder,
    band_start=band_start_ch,
    band_end=band_end_ch,
    vmin=VMIN,
    vmax=VMAX,
)

overlay_file = resp_create["output_file"]
print("Created:", overlay_file)
print("URL:", resp_create["output_url"])

sample_viewer_url = f"https://nucleonics.mit.edu/surface-viewer/?dataset={quote_plus(dataset)}"
resp_create["sample_viewer_url"] = sample_viewer_url
print("Sample viewer:", sample_viewer_url)

```

```python id="F7vc6pANHtR2"
# Show the current overlay filename for quick reuse or copy/paste.

overlay_file
```

```python id="36jZsLOhHfvW"
# Specify an overlay filename if you want to delete an existing overlay.

overlay_file_to_be_deleted = "overlays/heatmap_ch400-700_rng3-9039.json"  # replace with the file you want to delete
```

```python id="28976f42"
# # Delete the selected overlay heatmap file from the server.

# resp_delete = delete_overlay(
#     auth=auth,
#     dataset=dataset,
#     overlay_file=overlay_file_to_be_deleted,
# )
```

```python id="eRsaOwHggVev"

```
