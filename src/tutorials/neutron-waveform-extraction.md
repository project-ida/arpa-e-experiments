---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="7e269f27-20d0-42e1-a5eb-38e24a1c1b3e" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/neutrons/tutorials/neutron-waveform-extraction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/neutrons/tutorials/neutron-waveform-extraction.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="XXrIWBJj32kW" -->
# Radiation Detection Template for Waveform Extraction
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1s4u8Qh58Jix" outputId="ff3a1a00-e199-413e-f233-50ada03bcebe"
!pip install uproot
```

```python id="BzdI2BQzSkX-"
import uproot
import matplotlib.pyplot as plt
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # For log normalization
```

<!-- #region id="FVNb-Hl14kJs" -->
### Step  1 - Root file analysis for waveform extraction

Below we are following the steps to collect the waveform we are interested in looking at from the corresponding root file.

1. Identify time range of interest in the data panel
2. Aggregate and export corresponding .root files from Google Drive storage
3. Download aggregated files
4. Collab analysis!
<!-- #endregion -->

```python id="efrqRzrB91V2"
'''
Define thresholds for event selection
 psd_threshold: Pulse shape discrimination threshold for neutron identification
 energy_threshold: Energy threshold for event selection
'''

psd_threshold = 0.24
energy_threshold = 260
```

```python colab={"base_uri": "https://localhost:8080/"} id="Wa0A0iT-8Hnc" outputId="ead2d07c-ea5b-4379-8192-dbd4419e7e28"
file_path = "http://nucleonics.mit.edu/scripts/mergedrootfiles/Computers-thinkpad-t480s-NaI-RAW_20250305_1600_to_20250305_1630/merged_CH0_20250305_1600_to_20250305_1630.root"

file = uproot.open(file_path)

tree = file["Data_R"]  # Make sure this matches the name of the tree in your .root file

file_keys = file.keys()
file_keys
```

```python colab={"base_uri": "https://localhost:8080/"} id="IIRHAxiUTamb" outputId="72df186d-2728-4a2f-c9b8-29a9615601f1"
# Access the "Data_R" tree and list its branches
branches = tree.keys()
branches
```

```python id="0vL4v7t48nUN"
# Specify the leaves you want to extract
keys = ["Timestamp", "Energy", "EnergyShort"]
df = tree.arrays(keys, library="pd")  # Cast the leaves into a pandas DataFrame

# Convert Timestamp from picoseconds to seconds
df["Timestamp"] = df["Timestamp"] / 1e12

# Calculate the PSD value
df["PSD"] = (df["Energy"] - df["EnergyShort"]) / df["Energy"]
```

<!-- #region id="vYvDvTYmWFcN" -->
### Step 2 - Seperating Gamma and Neutron Events with PSD threshold
<!-- #endregion -->

<!-- #region id="JnqOMesP4rAD" -->
We plot PSD against ADC channel to distinguish neutron counts from background signals. The red dahsed line in the plot corresponds to the Pulse Shape Discrimination (PSD) threshold we defined earlier. We will consider counts above the line as Neutron counts and those below the line as gamma events.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="4WU55LeNTv9X" outputId="a75e3522-3248-4cb4-c658-1bb2a0fdd5b3"
# Create the 2D histogram
plt.figure(figsize=(10, 6))
H, energy_bins, psd_bins = np.histogram2d(df["Energy"], df["PSD"], bins=(512, 100), range=((0, 4000), (0, 1)))

# Plot the histogram with log normalization
plt.pcolormesh(energy_bins, psd_bins, H.T, norm=mcolors.LogNorm(), cmap='viridis', shading='auto')
plt.colorbar(label='Counts')

# Add a horizontal line at the desired PSD value
plt.axhline(y=psd_threshold, color='red', linestyle='--', linewidth=1)  # Modify y, color, linestyle, and linewidth as needed

# Label the axes
plt.xlabel("ADC channel (light output)")
plt.ylabel("PSD")
plt.title("2D Histogram of PSD vs ADC Channel")
plt.show()
```

<!-- #region id="FOkFKjcK7Kul" -->
Now with this information, we will keep only the desireable "neutron-like" pulses i.e. where the PSD exceeds the threshold.  
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="twmwRzWk8vL8" outputId="172f2ec2-2e34-442c-e624-4bcdb243ea7e"
# Filter the DataFrame to keep only pulses with PSD > threshold
filtered_df = df[df["PSD"] > psd_threshold]

# Check the first few rows of the filtered DataFrame
filtered_df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="uwSo8Exp-I30" outputId="5221d808-58ac-4be7-8f89-9d900e853eb0"
# Filter the DataFrame to keep only pulses with energy energy_threshold
filtered_df = filtered_df[filtered_df["Energy"] > energy_threshold]

# Check the first few rows of the filtered DataFrame
filtered_df
```

<!-- #region id="bqZEOxjSWPDF" -->
### Step 3 - Time History of Neutron Bursts/ Identifying time-region of interest
<!-- #endregion -->

<!-- #region id="heI6LpBTVtYT" -->
We now want to better identify when we observed a neutron burst. In order to do this, we will create a time history plot of neutron counts per second.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 584} id="i8knTHEM8vSL" outputId="c8832a62-5f23-4129-f405-0bdbc0118e51"
# To create a time history of counts per second, we will first bin the timestamps by each second
# and then count the number of pulses in each bin

# Bin the timestamps by second
time_bins = np.arange(df["Timestamp"].min(), df["Timestamp"].max() + 1, 1)  # Create 1-second bins
counts_per_second, _ = np.histogram(filtered_df["Timestamp"], bins=time_bins)

# Plot the counts per second
plt.figure(figsize=(10, 6))
plt.plot(time_bins[:-1], counts_per_second, marker='o', linestyle='-')
plt.title("Counts per Second Time History (PSD > 0.24)")
plt.xlabel("Time (s)")
plt.ylabel("Counts per Second")
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.show()
```

<!-- #region id="DlTmN8LXWsHh" -->
### Step 4 - Look into energy spectrum of filtered pulses

We are now interested in visualizing the distribution of detected energy values (i.e. those above the PSD threshold). In this way, we can check that what we have charactiereized as neutron-like counts show a characteristic neutron energy distribution.

Potential reference spectrum:
- Broad distribution with a tail
- Low energy cutoff
- no photopeaks (gamma)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="MbTNctMq8vVG" outputId="e40a2213-77b3-4b4c-e6ab-c667e5b582b7"
# To plot the energy spectrum of the filtered pulses
plt.figure(figsize=(10, 6))
plt.hist(filtered_df["Energy"], bins=100, range=(0, 4000), color='blue', alpha=0.7)
plt.title("Energy Spectrum (PSD > 0.24)")
plt.xlabel("Energy (ADC channel)")
plt.ylabel("Counts")
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.show()
```

<!-- #region id="FQLDEeuAYZ0V" -->
### Step 5 - Plot waveforms of neutron counts in burst region of interest
<!-- #endregion -->

```python id="VrPomPKS8vYE"
# Extract the indices of the pulses that meet the condition PSD > 0.24
filtered_indices = filtered_df.index
```

```python id="ss5jeAAvSN7O"
# Define the new time window
# We are now focusing on the spcific event cluster time region
time_window_center = 670200
time_window_start = time_window_center - 20
time_window_end = time_window_center + 15

plot_xlim_start = 670000
plot_xlim_end = 670500
```

<!-- #region id="VamoZ4evYEAu" -->
We have now identified a time-region of interest and filtered pulses within this time-window. We want to further analyze these pulses. In order to do so , we will extract and plot the corresponding waveforms.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 843} id="BpNxMcuTF82d" outputId="a2766e0d-c92e-4575-c001-dcebe35aae22"
# Bin the timestamps by second
time_bins = np.arange(df["Timestamp"].min(), df["Timestamp"].max() + 1, 1)  # Create 1-second bins
counts_per_second, _ = np.histogram(filtered_df["Timestamp"], bins=time_bins)

# Plot the counts per second
plt.figure(figsize=(10, 6))
plt.plot(time_bins[:-1], counts_per_second, marker='o', linestyle='-', label="Counts per Second")

# Add a semi-transparent rectangle for the time window
plt.axvspan(time_window_start, time_window_end, color='blue', alpha=0.1, label="Time Window")

# Apply x-axis limits
plt.xlim(plot_xlim_start, plot_xlim_end)

plt.title("Counts per Second Time History (PSD > 0.24)")
plt.xlabel("Time (s)")
plt.ylabel("Counts per Second")
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# Filter the pulses within the time window
pulses_in_window = filtered_df[(filtered_df["Timestamp"] >= time_window_start) & (filtered_df["Timestamp"] <= time_window_end)]

filtered_indices = pulses_in_window.index

# Extract only the waveforms corresponding to filtered_indices
selected_waveforms = tree["Samples"].array(entry_start=min(filtered_indices), entry_stop=max(filtered_indices) + 1)

# Plot the waveforms for the filtered pulses
plt.figure(figsize=(10, 6))
for idx in filtered_indices:
    relative_idx = idx - min(filtered_indices)  # Adjust index to match the sliced array
    plt.plot(selected_waveforms[relative_idx], alpha=0.5)

plt.title("Waveforms for Pulses with PSD > 0.24")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.ylim(9000, 17000)
plt.show()
```
