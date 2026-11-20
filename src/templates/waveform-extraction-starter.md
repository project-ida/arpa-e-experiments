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
- Access to the Nucleonics waveform server at `http://nucleonics.mit.edu:8080`, with the ROOT download route installed

ROOT files are downloaded through Nucleonics from the Synology mount. Google Drive is used only for SQL credentials.
<!-- #endregion -->

<!-- #region id="oG6NQb5BBF0p" -->
# Waveform extraction

This notebook contains starter code that extracts and visualises event waveforms from a ROOT file.
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
# Install the ROOT reader and HTTP download client
%pip install -q uproot requests

# SQL credentials are still loaded from Google Drive
import shutil
from google.colab import drive

# Data analysis and plotting
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt

# Our HTTP clients
from libs.sql_api import SQLConnection, SQLAPIError, read_sql
from libs.root_files_api import download_root_file
```

<!-- #region id="C99IMdA1BSWv" -->
## Authentication
<!-- #endregion -->

```python id="FhhgoaA3A0-Z"
# Mount Drive only to load SQL credentials; ROOT files now live on Synology.
drive.mount('/content/drive')
shutil.copy(
    "/content/drive/MyDrive/Nucleonics/.env/psql_credentials_readonly.py",
    "psql_credentials.py",
)
```

```python id="R5pArHA9BUJV"
# Import SQL credentials
from psql_credentials import PGHOST, PGUSER, PGPASSWORD, PGPORT, PGDATABASE

# Create the database connection string
connection_uri = f'https://{PGHOST}/sql-api/query.php'
engine = SQLConnection(
    connection_uri,
    PGUSER,
    PGPASSWORD,
    database=PGDATABASE,
    timeout=(10, 600),
)
```

<!-- #region id="uNJuPTnlBf3C" -->
## Fetch the ROOT file corresponding to the a time period
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
    df = read_sql(query, con=engine, params={
        "event_time": event_time,
        "file_pattern": file_pattern
    })
    if df.empty:
        raise FileNotFoundError(f"No indexed ROOT file found after {event_time} for channel {channel_number}.")
    df.set_index('time', inplace=True)
    return df
```

```python id="MULgCJIACxWN"
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
Now download the original ROOT file through the Nucleonics waveform server. It looks up the same database row and serves the file from its Synology mount.

The download is saved under `downloads/root_files` in this runtime. No waveform-server password is required. If the server returns 404, confirm that its ROOT download route has been deployed and that the file is available on Synology.
<!-- #endregion -->

```python id="nqUBfVWiJTg4"
ROOT_FILES_BASE_URL = "http://nucleonics.mit.edu:8080"
DOWNLOAD_DIR = "downloads/root_files"


def show_download_progress(received, total):
    downloaded_mb = received / 1024**2
    if total:
        print(f"Download {100 * received / total:.1f}% ({downloaded_mb:.1f} MB)", end="\r")
    else:
        print(f"Downloaded {downloaded_mb:.1f} MB", end="\r")
```

```python id="7kMHw24YMlGq"
print(f"Downloading {closest_root_file['file']} from Nucleonics...")
```

```python id="EgO6zwoTMlzN"
filename = download_root_file(
    closest_root_file,
    download_dir=DOWNLOAD_DIR,
    base_url=ROOT_FILES_BASE_URL,
    progress=show_download_progress,
)
print(f"\nSaved to {filename}")
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

```python id="k-Q12sRwptwP"
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

```python id="_erVbBxS9Bje"
time = pd.to_datetime(time)
print(time)
```

```python id="HEQExazK5W_U"
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

```python id="MmL3bAZs4jTz"
print(time)
```

<!-- #region id="F9RaTGIW4nXA" -->
with a duration of:
<!-- #endregion -->

```python id="U6Yl3U8K4sDE"
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

```python id="3XbaW-nClas1"
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

```python id="uSxZYs4blcjF"
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
