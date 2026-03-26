---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="4c47888e-cbe8-4640-8fa0-d976bed3542f" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/templates/waveform-extraction-starter-local.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/templates/waveform-extraction-starter-local.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="23ce122c-c3b5-4a7f-a8bb-448d8042509a" -->
# Waveform extraction (local / Synology version)

<!-- #endregion -->

<!-- #region id="ydG__I6QBDGM" -->

‼️ **Prerequisites** ‼️:
- The Synology share is mounted at `/mnt/synology/Computers`
- The PostgreSQL credentials file is available in your home directory as either `psql_credentials.py` or `psql_credentials_readonly.py`

<!-- #endregion -->

<!-- #region id="oG6NQb5BBF0p" -->
# Waveform extraction

This notebook contains starter code that extracts and visualises event waveforms from a ROOT file.
<!-- #endregion -->

<!-- #region id="Fu4x4NPbBKrT" -->
## Libraries
<!-- #endregion -->

```python id="IAyoM46dBFqg"
# Local notebook setup
import sys
import importlib
from pathlib import Path

# Data analysis
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sqlalchemy import create_engine, text
from scipy.stats import kstest
! pip install uproot -q
import uproot

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

```

<!-- #region id="C99IMdA1BSWv" -->
## Local setup

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 53} id="FhhgoaA3A0-Z" outputId="6e6348c1-23ef-471e-f3dd-2c6a3fb69c42"
# Configure local paths
HOME_DIR = Path.home()
SYNOLOGY_ROOT = Path("/mnt/synology/Computers")

if not SYNOLOGY_ROOT.exists():
    raise FileNotFoundError(
        f"Synology mount not found at {SYNOLOGY_ROOT}. "
        "Please confirm the NAS is mounted before running the notebook."
    )

if str(HOME_DIR) not in sys.path:
    sys.path.insert(0, str(HOME_DIR))

credentials_module_name = None
for candidate in ("psql_credentials", "psql_credentials_readonly"):
    if (HOME_DIR / f"{candidate}.py").exists():
        credentials_module_name = candidate
        break

if credentials_module_name is None:
    raise FileNotFoundError(
        "Could not find psql_credentials.py or psql_credentials_readonly.py "
        f"in {HOME_DIR}."
    )

```

```python id="R5pArHA9BUJV"
# Import SQL credentials from the user's home directory
credentials = importlib.import_module(credentials_module_name)
PGUSER = credentials.PGUSER
PGPASSWORD = credentials.PGPASSWORD
PGHOST = credentials.PGHOST
PGPORT = credentials.PGPORT
PGDATABASE = credentials.PGDATABASE

# Create the database connection string
connection_uri = f'postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}'
engine = create_engine(connection_uri)

```

<!-- #region id="uNJuPTnlBf3C" -->
## Locate the ROOT file corresponding to a time period

<!-- #endregion -->

<!-- #region id="BXAl9Bq4BMNs" -->
For this notebook we must specify a:
- channel number
- time of interest
- duration around the time of interest
<!-- #endregion -->

```python id="9oSGSyO0A_QD"
channel_number = 6
time = '2025-09-03 12:00:00'
duration = 600
```

<!-- #region id="PX5eyO2WJg8T" -->
We now look in the database for the closest ROOT file to the `time`. Since our ROOT files are organised by their end time (time of the last event), we need to look for the closest ROOT file after our specified `time`.
<!-- #endregion -->

```python id="hXNM2rAOBYlH"
def find_root_file(event_time, channel_number):
    query = """
    SELECT *
    FROM root_files
    WHERE time > %(event_time)s
      AND file LIKE %(file_pattern)s
    ORDER BY ABS(EXTRACT(EPOCH FROM (%(event_time)s - time)))
    LIMIT 1;
    """
    file_pattern = f"%_CH{channel_number}@%"
    df = pd.read_sql(query, con=engine, params={
        "event_time": event_time,
        "file_pattern": file_pattern
    })
    df.set_index('time', inplace=True)
    return df

```

```python colab={"base_uri": "https://localhost:8080/", "height": 112} id="MULgCJIACxWN" outputId="27830e06-d946-4a8d-a1fc-a395b202cf81"
df_root = find_root_file(time, channel_number)
df_root
```

<!-- #region id="2QLmFKCFN7UM" -->
Let's create a dictionary for easy access to the file info:
<!-- #endregion -->

```python id="nqjenrX_PLns"
closest_root_file = df_root.iloc[0].to_dict()
```

<!-- #region id="capsaKe7JnX-" -->
Now we construct the local Synology-mounted path for this ROOT file so that we can open it directly with `uproot`.

<!-- #endregion -->

```python id="nqUBfVWiJTg4"
def construct_file_path(root_file):
    return str(
        SYNOLOGY_ROOT / root_file["computer"] / root_file["dir"] / root_file["file"]
    )

```

```python id="O0d0eZmvJUgK"
file_path = construct_file_path(closest_root_file)
file_path

```

```python colab={"base_uri": "https://localhost:8080/"} id="EgO6zwoTMlzN" outputId="5b9329a5-9f07-4010-b952-8c06897fd23a"
filename = file_path

if not Path(filename).exists():
    raise FileNotFoundError(f"ROOT file not found: {filename}")

print(f"Using local ROOT file: {filename}")

```

<!-- #region id="iAyWI3QRlSIM" -->
## Analysing waveform data

### Pulses with absolute time

Each radiation pulse is stored in the ROOT files with a `Timestamp`, `Energy`, `EnergyShort` and `Samples`. The waveform of the pulse is stored in the `Samples` part of the ROOT tree.

`Timestamp` information measures the number of picoseconds from the start of the experiment. In order for us to conveniently extract pulses using absolute time, we will need to create a dataframe with the appropriate time index.
<!-- #endregion -->

```python id="1zfkD_z3Fhmw"
# Opens up the ROOT file and extracts the information
with uproot.open(filename) as f:
    tree = f['Data_R']
    branches = tree.keys()
    ts = tree['Timestamp'].array(library='np')
    e = tree['Energy'].array(library='np')
    wf = tree['Samples'].array(library='np')

    # Conditionally load EnergyShort if it exists
    if 'EnergyShort' in branches:
        es = tree['EnergyShort'].array(library='np')
    else:
        es = None

# Creates dataframe with absolute time index
end_time = df_root.index[0]
start_time = end_time - pd.to_timedelta((ts[-1]-ts[0])/1e12, unit='s')
time_abs = start_time + pd.to_timedelta((ts - ts[0])/1e12, unit='s')

data = {'Energy': e, 'Waveform': wf}
if es is not None:
    data['EnergyShort'] = es
df_pulses = pd.DataFrame(data, index=pd.to_datetime(time_abs))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="k-Q12sRwptwP" outputId="2d8e8a16-6e34-4be1-9a5a-810eabcb6190"
df_pulses.head()
```

<!-- #region id="WbmLZnb7p2LP" -->
### Filtering pulses

We don't necessarily want to see all pulses. We might wish to only see high energy pulses, or filter out neutrons from gammas (if using an Eljen detector) using the PSP value (More discussion about neutron/gamma discrimination can be found in the [PSD Analysis notebook](https://github.com/project-ida/arpa-e-experiments/blob/main/tutorials/PSD-analysis.ipynb)).

PSP is calculated by:

$$
\rm PSP = \frac{\rm Energy - EnergyShort}{\rm Energy}
$$


<!-- #endregion -->

```python id="XqqcpWDg3d0g"
if "EnergyShort" in df_pulses.columns:
  df_pulses['PSP'] = 1 - (df_pulses['EnergyShort'] / df_pulses['Energy'])
```

```python id="bTRM1Zt3CbKz"
psp_range = [0,1] # All PSP values

energy_range = [0,70] # low energy
# energy_range = [71,600] # high energy
# energy_range = [901,3100] # very high energy
```

```python id="pYmh_1b3lYmQ"
filtered = df_pulses[(df_pulses['Energy'] > energy_range[0]) & (df_pulses['Energy'] < energy_range[1])]

if "PSP" in df_pulses.columns:
  filtered = filtered[(filtered['PSP'] > psp_range[0]) & (filtered['PSP'] < psp_range[1])]
```

<!-- #region id="KlFWvHRFrxS7" -->
### Visualising the counts

Let's now do a sanity check and visualise the counts per second over the entire ROOT file. The plot below should look identical to what we see in our data panels.

We'll also mark the burst time:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_erVbBxS9Bje" outputId="cfea3281-da75-4ee6-c723-9f23a02ed483"
time = pd.to_datetime(time)
print(time)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="HEQExazK5W_U" outputId="908bd92e-418c-4973-9a80-d19a4a54548e"
cps = filtered["Energy"].resample('1s').count()
plt.figure(figsize=(12, 4))
plt.plot(cps, drawstyle='steps-mid')
plt.axvline(time, color='r', linestyle='--', label="Time of interest")
plt.title('Full ROOT file')
plt.xlabel('Time')
plt.ylabel('Counts per second')
plt.legend()
plt.grid(True)
plt.show()
```

<!-- #region id="NbZfwRbD5oBS" -->
Let's now zoom in on the time period of interest which is around:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MmL3bAZs4jTz" outputId="cc8ff7b2-eab3-4393-a444-142e82767215"
print(time)
```

<!-- #region id="F9RaTGIW4nXA" -->
with a duration of:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="U6Yl3U8K4sDE" outputId="a3072ea5-0d3e-4da7-80c2-3275c52d2509"
duration = pd.to_timedelta(duration, unit='s')
print(duration)
```

<!-- #region id="WAdMOtV85BqZ" -->
Once we determine the `start` and `end` times, we can easily pull out cps and waveforms for the burst.
<!-- #endregion -->

```python id="8xyr7GiE3bV9"
start = time - duration/2
end = time + duration/2
```

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="3XbaW-nClas1" outputId="aae10569-f911-4af1-d612-32447f0a4392"
plt.figure(figsize=(12, 4))
plt.plot(cps[start:end], drawstyle='steps-mid', color='orange')
plt.axvline(time, color='r', linestyle='--', label="Time of interest")
plt.title('Period of interest')
plt.xlabel('Time')
plt.ylabel('Counts per second')
plt.legend()
plt.grid(True)
plt.show()
```

<!-- #region id="Sy9aHwBy24Co" -->
### Visualising the waveforms

We can now easily extract and visualise the waveforms using the `start` and `end` times.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 606} id="uSxZYs4blcjF" outputId="eb074715-2abe-4b18-d570-ebc5c0c7cb3a"
df_period = filtered[start:end]

plt.figure(figsize=(10, 6))
for wf in df_period['Waveform']:
    plt.plot(wf, alpha=0.5)

if "EnergyShort" in df_pulses.columns:
  plt.title(f'Waveforms for pulses between {start} and {end} \n with PSP in range {psp_range}\n and energy in range {energy_range}')
else:
  plt.title(f'Waveforms for pulses between {start} and {end} \n with energy in range {energy_range}')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

```python id="h7F_gl3C_7Ee"

```
