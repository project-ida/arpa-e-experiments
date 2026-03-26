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

<!-- #region id="7daa6b8c-7515-4843-b533-3cbed9054895" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/templates/eds-sample-analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/templates/eds-sample-analysis.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="7f188c81" -->
# EDS Analysis template

This notebook provides a quick, end-to-end demo for ROI-based EDS analysis:
it pulls saved regions from the Surface Viewer API, builds a table of selected cells, loads the corresponding spectra, aggregates them, and runs a simple first-pass peak analysis.

## How to use

1. Add the `api_url` (get the link from a surface viewer selection grid)
name.
2. Run cells top-to-bottom.
3. Use the aggregate plot and peak table as a rapid “first look” before moving to the full testbed.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_K48SSZ_Z44M" outputId="adc4a3a4-a1b0-4bec-95c2-7a3589de4c0c"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 635} id="9df52624" outputId="390bf89d-e544-45d4-90dd-674ee2a29c2c"
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

```python colab={"base_uri": "https://localhost:8080/"} id="f7c5adb5" outputId="f0743f30-8ec6-4300-af40-0b9d200acae5"
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

```python colab={"base_uri": "https://localhost:8080/"} id="83346e85" outputId="c26afe41-dcb8-4aa1-d214-84dd4b63e41b"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="6653d088" outputId="a3439910-1e28-4ee6-db48-a8cac452f1aa"
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

```python colab={"base_uri": "https://localhost:8080/"} id="b97f6ba8" outputId="b7c4030f-0ea7-4003-b688-208ac9a987c6"
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

```python colab={"base_uri": "https://localhost:8080/"} id="0ef9187d" outputId="c94f2e99-8311-4fb3-d273-29c287181fbc"
# Report which energy calibration was found and whether config values or defaults are being used.

if cal is not None:
    source_text = "config.txt" if cal.get("from_config") else "defaults"
    print(f"Energy calibration source: {source_text}")
    print("Calibration:", cal)
else:
    print("No energy calibration available; top energy axis will be omitted.")

```

```python colab={"base_uri": "https://localhost:8080/", "height": 506} id="b2b49706" outputId="1277fbd6-fb55-4ae6-b45a-ce0de75efac3"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="c9ae05a3" outputId="1ece6679-0054-44de-d75d-a24419234216"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="WyWhFSHHjWzj" outputId="4cd7ca4d-ac37-4b47-b0db-9f6eabc0e99d"
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

```python colab={"base_uri": "https://localhost:8080/"} id="31a290de" outputId="5360f884-aa94-404a-a236-cf3e6c04b4c0"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 967} id="12d03a2b" outputId="cce3ddcb-6ebb-4a0b-e74e-5c619afe6c40"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="c1c277c6" outputId="99bbacba-825e-4ad8-878b-0c270eb0c8e5"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 332} id="a746c947" outputId="926384a7-1a03-4244-e635-9f25000bc731"
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
VMIN = 0
VMAX = 11262
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="5tVt5_NzI30V" outputId="bdf66289-43e4-4f15-aaaa-1b8595febaf4"
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

```python colab={"base_uri": "https://localhost:8080/"} id="33caf51e" outputId="23661f4e-35a0-4419-ce7d-35568f51609b"
# Authenticate to the overlay API and extract dataset metadata from the ROI URL.

auth = get_api_auth()

dataset = parse_qs(urlparse(ROI_API_URLS[0]).query)["dataset"][0]
input_folder = "aggregated-spectra"

print("Dataset:", dataset)
print("Input folder:", input_folder)
print(f"Using resolved band: {band_start_ch}–{band_end_ch} channels")
```

```python colab={"base_uri": "https://localhost:8080/"} id="599dffb5" outputId="5cbb4eef-2494-4ea5-8513-fdf711fbccab"
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

```python id="F7vc6pANHtR2" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="986fe75b-a332-4f1b-cf2e-0eafac9827c5"
# Show the current overlay filename for quick reuse or copy/paste.

overlay_file
```

```python id="36jZsLOhHfvW"
# Specify an overlay filename if you want to delete an existing overlay.

overlay_file_to_be_deleted = "overlays/heatmap_ch500-1000_rng9000-12000.json"  # replace with the file you want to delete
```

```python id="28976f42"
# # Delete the selected overlay heatmap file from the server.

# resp_delete = delete_overlay(
#     auth=auth,
#     dataset=dataset,
#     overlay_file=overlay_file_to_be_deleted,
# )
```
