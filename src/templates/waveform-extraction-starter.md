---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="23ce122c-c3b5-4a7f-a8bb-448d8042509a" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/templates/waveform-extraction-starter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/templates/waveform-extraction-starter.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="ydG__I6QBDGM" -->
‼️ **Prerequisites** ‼️:
- Access to the `Nucleonics` Google Drive shortcut and its `.env` folder for SQL credentials
- Access to Nucleonics data

ROOT files are downloaded from Nucleonics and saved locally for analysis. Google Drive is used only for SQL credentials.
<!-- #endregion -->

<!-- #region id="oG6NQb5BBF0p" -->
# Waveform extraction

Choose a detector channel and time interval, then load its pulses into a pandas DataFrame for filtering and plotting. The waveform library handles finding, downloading, and combining the relevant ROOT files.
<!-- #endregion -->

```python id="iRx5OjNEHaKe"
# Makes Libs accessible and runs notebook from same location regardless of whether colab or local
!pip install colocal -q
import colocal
root, branch, cwd = colocal.setup()
```

<!-- #region id="Fu4x4NPbBKrT" -->
## Libraries
<!-- #endregion -->

```python id="IAyoM46dBFqg"
%pip install -q uproot requests

import shutil
from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt

from libs.sql_api import SQLConnection
from libs.waveforms import load_waveforms, read_waveforms
```

<!-- #region id="C99IMdA1BSWv" -->
## SQL connection

The ROOT files table (`root_files`) is queried through our SQL API. Load the SQL credentials and create a connection here; `load_waveforms()` receives that connection and uses it to find the files for the requested detector and interval.
<!-- #endregion -->

```python id="FhhgoaA3A0-Z"
# Mount Drive to load SQL credentials.
drive.mount('/content/drive')
shutil.copy(
    "/content/drive/MyDrive/Nucleonics/.env/psql_credentials_readonly.py",
    "psql_credentials.py",
)
```

```python id="R5pArHA9BUJV"
from psql_credentials import PGHOST, PGUSER, PGPASSWORD, PGDATABASE

connection = SQLConnection(
    f"https://{PGHOST}/sql-api/query.php",
    PGUSER,
    PGPASSWORD,
    database=PGDATABASE,
    timeout=(10, 600),
)
```

<!-- #region id="uNJuPTnlBf3C" -->
## Choose the detector and time interval

Specify the device (for example, `caen8ch`), channel, time of interest, and duration around it. You can also assign `start` and `end` directly. The interval includes `start` and excludes `end`.
<!-- #endregion -->

```python id="9oSGSyO0A_QD"
device = "caen8ch"
channel_number = 6
time_of_interest = pd.Timestamp("2025-09-03 12:00:00")
duration = pd.Timedelta(minutes=10)

start = time_of_interest - duration / 2
end = time_of_interest + duration / 2
print(f"Selected interval: {start} to {end}")
```

<!-- #region id="PX5eyO2WJg8T" -->
## Load the pulses

ROOT is a binary data format that can store compressed measurement data. In these files, the `Data_R` tree contains one entry per pulse:

| Field | Meaning |
|---|---|
| `Timestamp` | Integer picoseconds elapsed since acquisition start |
| `Energy` | Long-gate integrated pulse value, used as an energy measure |
| `EnergyShort` | Short-gate integrated value, when recorded |
| `Samples` | Waveform amplitudes for the pulse |

`Timestamp` alone does not give calendar time: it measures time relative to acquisition start. The ROOT files table stores the absolute time of each file's last pulse in its `time` column. `load_waveforms()` uses this value and the relative ROOT timestamps to calculate the absolute time of every pulse.

The function finds and downloads the relevant ROOT files, combines their pulses, and returns a DataFrame containing only the requested interval. The original `Timestamp` values are retained.

The times in the ROOT files table omit fractional seconds, so reconstructed absolute times can differ from the original pulse-table times by a fraction of a second.
<!-- #endregion -->

```python id="MULgCJIACxWN"
df_pulses = load_waveforms(
    connection,
    device=device,
    channel=channel_number,
    start=start,
    end=end,
)

print(f"Loaded {len(df_pulses)} pulses from {len(df_pulses.attrs['root_files'])} files.")
df_pulses.head()
```

<!-- #region id="b2c36603c32e43bca08a4b5aa84f7beb" -->
The DataFrame is indexed by reconstructed absolute pulse time. `Waveform` contains the sample arrays, and `SourceFile` identifies where each pulse came from. `EnergyShort` and the calculated `PSP` column are included when available.

### Reuse files already downloaded

The downloaded files remain available for further analysis. To read them again, use their paths and per-file end times saved in `df_pulses.attrs["root_files"]`:

```python
saved_files = pd.DataFrame(df_pulses.attrs["root_files"])
df_pulses = read_waveforms(saved_files, start=start, end=end)
```

`read_waveforms()` can also take a list of local file paths, deriving their end times from the processed filenames. Omitting `start` and `end` reads all pulses in those files. See `libs/README.md` for saving the file list for a later session.
<!-- #endregion -->

<!-- #region id="WbmLZnb7p2LP" -->
## Analyse the pulses

### Filter by energy and PSP

We can select pulses by energy and, when `EnergyShort` is available, by the pulse-shape parameter PSP. This is useful for neutron/gamma discrimination with Eljen detectors; see the [PSD Analysis notebook](https://github.com/project-ida/arpa-e-experiments/blob/main/tutorials/PSD-analysis.ipynb).

The library calculates:

$$
\rm PSP = \frac{\rm Energy - EnergyShort}{\rm Energy}
$$

Set either range to `None` to disable that selection. The bounds below are exclusive. PSP is `NaN` when the required short-gate value is missing or energy is zero; applying a PSP range excludes those pulses.
<!-- #endregion -->

```python id="pYmh_1b3lYmQ"
energy_range = (0, 70)  # Low energy; try (71, 600) or (901, 3100)
psp_range = None         # For example, (0.237, 1.0) for a PSP selection

filtered = df_pulses
if energy_range is not None:
    filtered = filtered[
        (filtered["Energy"] > energy_range[0]) &
        (filtered["Energy"] < energy_range[1])
    ]
if psp_range is not None:
    if "PSP" not in filtered.columns:
        raise ValueError("PSP selection requires EnergyShort data.")
    filtered = filtered[
        (filtered["PSP"] > psp_range[0]) &
        (filtered["PSP"] < psp_range[1])
    ]
print(f"Selected {len(filtered)} of {len(df_pulses)} pulses.")
```

<!-- #region id="KlFWvHRFrxS7" -->
### Counts during the requested interval

Plot the selected pulses in one-second bins, including bins with no pulses. The red line marks the time of interest. Bins at the edges may cover only part of a second.
<!-- #endregion -->

```python id="HEQExazK5W_U"
bins = pd.date_range(start.floor("s"), end.ceil("s"), freq="1s", inclusive="left")
cps = filtered["Energy"].resample("1s").count().reindex(bins, fill_value=0)

plt.figure(figsize=(12, 4))
plt.plot(cps, drawstyle="steps-mid")
plt.axvline(time_of_interest, color="r", linestyle="--", label="Time of interest")
plt.xlim(start, end)
plt.title("Selected pulses during the requested interval")
plt.xlabel("Time")
plt.ylabel("Counts per one-second bin")
plt.legend()
plt.grid(True)
plt.show()
```

<!-- #region id="Sy9aHwBy24Co" -->
### Waveforms during the requested interval

Overlay the waveforms for the selected pulses. Each sample array is plotted against its sample number; the DataFrame's time index identifies when that pulse occurred.
<!-- #endregion -->

```python id="uSxZYs4blcjF"
plt.figure(figsize=(10, 6))
for waveform in filtered["Waveform"]:
    plt.plot(waveform, alpha=0.5)

selection = f"Energy range: {energy_range}; PSP range: {psp_range}"
plt.title(f"{len(filtered)} waveforms from {start} to {end}\n{selection}")
plt.xlabel("Sample number")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
```
