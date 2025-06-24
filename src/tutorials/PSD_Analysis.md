---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/tutorials/PSD_Analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/tutorials/PSD_Analysis.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>

<!-- #region id="zPGvduxO5Zva" -->

‼️ **Prerequisites** ‼️:
- Access to the `Nucleonics` Google drive folder (it must also be added as a shortcut in your own drive)
- Access to the nucleonics `.env` folder (where sensitive info lives)
<!-- #endregion -->

<!-- #region id="40OAJwX8aE4K" -->
# PSD Analysis

The analysis notebook relies on the "Nuclear particle master" sheet to provide timestamps that allow us to separate the radiation data from experiments into several regions.
- Set-up
- Calibration (30 min with source)
- Background 1 (12 hours)
- Experiment
- Background 2 (12 hours)

A PSD plot will be created for the calibration period in order to extract a `psp_threshold` that allows us to discriminate between gammas and neutrons. This threshold is then saved back to the master sheet.


<!-- #endregion -->

<!-- #region id="WLRzIEVx2E1h" -->
## Running this notebook

Go ahead and change the `experiment_id` below and then run the whole notebook.

You will be asked a couple of time to authenticate with your Google account, but after that all the analysis will happen automatically.
<!-- #endregion -->

```python id="Fh2oRsfz2EKK"
experiment_id = 1
```

<!-- #region id="ngm6-Aeq3BBl" -->
## Libraries
<!-- #endregion -->

```python id="jBFPVytJarKY"
# Auth
import sys, os
import shutil
from google.colab import drive
from google.colab import auth
from google.auth import default

# Interacting with sheets
import gspread

# Data analysis
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sqlalchemy import create_engine, text

#Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
```

<!-- #region id="ECkL-FKb0z8d" -->
## Authentication

We need to do a few authentication steps:
- Bring in the database credentials from Google drive so that we can pull data from the live database.
- Bring in the nuclear particle master sheet ID
-  Authenticate Colab to pull the nuclear particle master sheet using the Drive API.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hLa3yxHiau8o" outputId="bdb45b31-49e7-4c05-a95d-24bc874a6d5d"
# Mount Drive
drive.mount('/content/drive')

# Copy SQL credentials from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/psql_credentials.py", "psql_credentials.py")

# Copy sheet ID file from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/sheet_ids.py", "sheet_ids.py");
```

```python id="yxzMFBara5ov"
# Import SQL credentials
from psql_credentials import PGUSER, PGPASSWORD, PGHOST, PGPORT, PGDATABASE

# Import sheet ID for the nuclear particle master sheet
from sheet_ids import NUCLEAR_PARTICLE_MASTER as sheet_id

# Create the database connection string
connection_uri = f'postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}'
engine = create_engine(connection_uri)
```

```python id="qFmXJoGKj_Sh"
# Authenticate using Colab's built-in credentials
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
```

<!-- #region id="ejMm7SW91R4A" -->
## Extracting experimental timestamps

We need to
- Open the master sheet
- Find the row corresponding with the experiment
- Extract the timestamp columns
<!-- #endregion -->

```python id="-COStQ5FkKf7"
sheet = gc.open_by_key(sheet_id).sheet1

# Read the sheet into a pandas DataFrame
df = pd.DataFrame(sheet.get_all_records())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="bfHvT6Qtkvbq" outputId="74cb0aeb-8497-4a96-c170-35ce67286627"

# Find the row where Experiment ID matches
row = df[df['Experiment ID'] == experiment_id]

# Extract times from columns M, N, O, P, Q, R, S
times = row[['Setup start', 'Calibration start', 'Background 1 start', 'Experiment start', 'Background 2 start', 'Background 2 end']]

times = times.apply(pd.to_datetime)

# Display the extracted times
times.head()
```

<!-- #region id="8l-cIG5C1vSF" -->
## Compiling PSD data from radiation events

We collect the time, energy and psp value of each radiation event and store it in a SQL database. In order to create a PSD plot and perform analysis, we need to work with 2D histogram. In other words, we need to count the number of events in given energy/psp buckets.

The database is optimised for performing these kind of aggregations over large volumes of data so we lean on its capabilities instead of attempting to bring all the data into python and then aggregating.

<!-- #endregion -->

```python id="aXUSKmFPlpfT"
def get_psd_data(start_time, end_time):
  query = f"""
  SELECT
      width_bucket(channels[1], 0, 1, 128) AS psp_bin,
      width_bucket(channels[2], 0, 4000, 512) AS energy_bin,
      COUNT(*) AS count
  FROM caen8ch_ch0
  WHERE time BETWEEN '{start_time}' AND '{end_time}'
  GROUP BY psp_bin, energy_bin
  ORDER BY psp_bin, energy_bin;
  """
  return pd.read_sql(query, engine, index_col=None)

def get_all_psd_data(times):
  psd_data = []
  for i in range(len(times.iloc[:,1:].values[0])):
    data = get_psd_data(times.iloc[0, i], times.iloc[0, i + 1])
    psd_data.append(data)
  return psd_data
```

<!-- #region id="6VVPzETa25RP" -->
We will now extract PSD data for all periods in our experiment.
<!-- #endregion -->

```python id="jkTzpa8ernu2"
calibration_data, background_1_data, experiment_data, background_2_data = get_all_psd_data(times.iloc[:,1:])
```

<!-- #region id="e-WFcA3Y3f6X" -->
## Visualising the PSD plots
<!-- #endregion -->

```python id="bHU6fOgQmTKj"
def plot_psd(data, title="PSD", psp_threshold=None, ax=None):
    # Initialise histogram
    hist = np.zeros((512, 128), dtype=int)

    # Accumulate counts
    for row in data.itertuples(index=False):
        psp_bin, energy_bin, count = row
        if 0 <= energy_bin < 512 and 0 <= psp_bin < 128:
            hist[energy_bin, psp_bin] += int(count)

    # Define bin edges
    energy_edges = np.linspace(0, 4000, 513)  # 512 bins → 513 edges
    psp_edges = np.linspace(0, 1, 129)        # 128 bins → 129 edges

    # Plot with pcolormesh and log colour scale
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

    im = ax.pcolormesh(energy_edges, psp_edges, hist.T,
                      norm=mcolors.LogNorm(vmin=1, vmax=hist.max()),
                      cmap='viridis', shading='auto')
    ax.set_xlabel('Energy')
    ax.set_ylabel('PSP')
    ax.set_title(title)

    # Draw horizontal red line at psp_threshold if provided
    if psp_threshold is not None:
        if 0 <= psp_threshold <= 1:  # Ensure threshold is within PSP range [0, 1]
            ax.axhline(y=psp_threshold, color='red', linestyle='-', label=f'PSP Threshold: {psp_threshold:.3f}')
            ax.legend()

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Count (log scale)')

    if ax is None:
        plt.show()
```

<!-- #region id="Zo96WMp-3jrF" -->
We begin with the calibration period for which we have the largest number of events due to the presence of a source of radiation. This PSD plot is what we'll use to extract a simple psp threshold value that can be used to quickly discriminate between gammas (lower psp) and neutrons (higher psp).  
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="GDJzrD8zmFV-" outputId="b2e4d91a-220a-4cf4-b78d-e3ffe184bcab"
plot_psd(calibration_data, f'Calibration (30 min)')

```

<!-- #region id="77Je4rGQ4dAt" -->
## Calculating the PSP threshold

The most accurate way to discriminate between gammas and neutrons is to create a fiducial lines. For the purpose of eyeballing neutron/gamma counts during a live experiment, a simple threshold psp value can be a good enough.

In this notebook, we trial a midpoint method.
- Choose an energy value
- Determine the psp locations of the gamma and neutron peaks
- Take the half way point between the two values.
<!-- #endregion -->

```python id="awexRLhIuuFq"
def find_psp_midpoint(data, target_energy=500, energy_range=(0, 4000), psp_range=(0, 1), energy_bins=512, psp_bins=128):
    # Step 1: Map target_energy to the closest energy_bin
    bin_width = (energy_range[1] - energy_range[0]) / energy_bins  # 4000 / 512 = 7.8125
    closest_energy_bin = int(round(target_energy / bin_width))
    closest_energy_bin = max(1, min(closest_energy_bin, energy_bins))  # Clamp to [1, 512]
    print(f"Closest energy bin to {target_energy}: {closest_energy_bin}")

    # Step 2: Filter data for the closest energy_bin
    filtered_data = data[data['energy_bin'] == closest_energy_bin]
    if filtered_data.empty:
        raise ValueError(f"No data found for energy bin {closest_energy_bin} (energy ~{target_energy})")

    # Step 3: Prepare PSP distribution for peak detection
    psp_bin_width = (psp_range[1] - psp_range[0]) / psp_bins  # 1 / 128 = 0.0078125

    # Create a histogram-like array for the full PSP range
    hist = np.zeros(psp_bins)
    for idx, row in filtered_data.iterrows():
        psp_bin = row['psp_bin'] - 1  # Convert to 0-based index (since psp_bin is 1 to 128)
        count = row['count']
        if 0 <= psp_bin < psp_bins:
            hist[psp_bin] += count

    # Find peaks using scipy.signal.find_peaks
    peaks_indices, _ = find_peaks(hist, height=0, prominence=10)  # Adjust prominence as needed
    if len(peaks_indices) < 2:
        raise ValueError("Not enough significant peaks detected; adjust prominence or check data.")

    # Get the two highest peaks based on prominence
    peak_indices_sorted = sorted(peaks_indices, key=lambda x: hist[x], reverse=True)[:2]
    if len(peak_indices_sorted) < 2:
        raise ValueError("Not enough prominent peaks to determine midpoint.")

    # Convert peak indices to PSP values
    peak1_idx, peak2_idx = peak_indices_sorted[:2]
    peak1_value = (peak1_idx + 0.5) * psp_bin_width  # Bin center
    peak2_value = (peak2_idx + 0.5) * psp_bin_width  # Bin center
    peak1_value, peak2_value = sorted([peak1_value, peak2_value])  # Ensure peak1 < peak2
    print(f"Peak PSP values: {peak1_value:.6f}, {peak2_value:.6f}")

    # Step 4: Calculate midpoint in PSP units
    midpoint = (peak1_value + peak2_value) / 2
    print(f"Midpoint between PSP peaks: {midpoint:.6f}")

    # Step 5: Plot PSP distribution with peaks and midpoint
    plt.figure(figsize=(8, 5))
    psp_centers = np.arange(psp_bins) * psp_bin_width + psp_bin_width / 2  # Bin centers
    plt.bar(psp_centers, hist, width=psp_bin_width, alpha=0.7)
    plt.axvline(peak1_value, color='r', linestyle='--', label=f'Peak 1: {peak1_value:.3f}')
    plt.axvline(peak2_value, color='g', linestyle='--', label=f'Peak 2: {peak2_value:.3f}')
    plt.axvline(midpoint, color='b', label=f'Midpoint: {midpoint:.3f}')
    plt.xlabel('PSP')
    plt.ylabel('Count')
    plt.title(f'PSP Distribution at Energy ~{target_energy}')
    plt.legend()
    plt.show()

    return midpoint
```

<!-- #region id="t3i4sq1N5oUB" -->
In this analysis we'll use Energy  = 500
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 541} id="eq9yNEu9u1WM" outputId="fc199b9a-38ea-489c-d570-fb56c957ac91"
target_energy = 500
psp_threshold = find_psp_midpoint(calibration_data)
```

<!-- #region id="k3oLzeKc53Y4" -->
Now that we have our threshold, we can remake the PSD plots to see if there is anything obviously wrong with the analysis.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="GDItxFMPu7xY" outputId="db4ef54f-1c74-4822-dcb5-e7a2434fc03a"
plot_psd(calibration_data, 'Calibration (30 min)', psp_threshold)

```

<!-- #region id="tIsjeWlh6EuO" -->
It can also be instructive to create the PSD plots for both background periods - we would not expect significant changes between the two. Here, we just eyeball them, but performing a statistical analysis will be the next step.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 487} id="lqeyLIBawd9O" outputId="cfe5fa34-3315-418d-bb96-5b4a9c549e3c"

fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
axes = axes.flatten()  # Flatten for easy indexing
plot_psd(background_1_data, title='Background 1 (12 hours)', psp_threshold=psp_threshold, ax=axes[0])
plot_psd(background_2_data, title='Background 2 (12 hours)', psp_threshold=psp_threshold, ax=axes[1])
```

<!-- #region id="X9aCY0ta6rdZ" -->
And of course we can look at the experimental period.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="0mYMvYpTsdf1" outputId="e2644bb3-b4a5-47d8-80d1-df12e86b8542"
experiment_period = (times.iloc[0]["Background 2 start"] - times.iloc[0]["Experiment start"]).days
plot_psd(experiment_data, f'Experiment ({experiment_period} days)', psp_threshold=psp_threshold)

```

<!-- #region id="x8mWf6Sky1SA" -->
Finally, we now update the master spreadsheet with the PSP threshold that will be used to gamma/neutron discrimination.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fPphtneprA95" outputId="825bb417-6d69-4c94-e848-3d7d024d2308"
# Update the DataFrame with the midpoint value
if not row.empty:
    row_index = df.index[df['Experiment ID'] == experiment_id][0]
    df.at[row_index, 'psp threshold'] = psp_threshold  # Update the 'psp threshold' column
    print(f"Updated 'psp threshold' to {psp_threshold:.6f} for Experiment ID {experiment_id}")

    # Clean the DataFrame to handle NaN values
    df_clean = df.fillna('')  # Replace NaN with empty string, or use 0 if appropriate

    # Write the updated DataFrame back to the Google Sheet using gspread
    values = [df_clean.columns.tolist()] + df_clean.values.tolist()  # Header + data
    sheet.update(values, 'A1')  # Update starting at A1
    print("Google Sheet updated successfully.")
else:
    print(f"No row found for Experiment ID {experiment_id}")
```
