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

<!-- #region id="23ce122c-c3b5-4a7f-a8bb-448d8042509a" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/templates/neutron-waveform-extraction-starter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/templates/neutron-waveform-extraction-starter.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="ydG__I6QBDGM" -->

‼️ **Prerequisites** ‼️:
- Access to the `Nucleonics` Google drive folder (it must also be added as a shortcut called "Nucleonics" in your own drive)
- Access to the nucleonics `.env` folder (where sensitive info lives)
<!-- #endregion -->

<!-- #region id="oG6NQb5BBF0p" -->
# Neutron waveform extraction

This notebook contains starter code that extracts and visualises neutron event waveforms from a ROOT file.
<!-- #endregion -->

<!-- #region id="Fu4x4NPbBKrT" -->
## Libraries
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IAyoM46dBFqg" outputId="d291f2bc-a8b8-4014-bdfe-f82a63682512"
# Auth
import sys, os
import shutil
from google.colab import drive
from google.colab import auth
from google.auth import default

# Files
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
import gspread

# Data analysis
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sqlalchemy import create_engine, text
from scipy.stats import kstest
! pip install uproot -q
import uproot

#Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
```

<!-- #region id="C99IMdA1BSWv" -->
## Authentication
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 53} id="FhhgoaA3A0-Z" outputId="6416ca94-529d-48a1-d234-43fc95312d39"
# Mount Drive
drive.mount('/content/drive')

# Authenticate and create the Drive API client
auth.authenticate_user()
drive_service = build('drive', 'v3')
creds, _ = default()
gc = gspread.authorize(creds)

# Copy SQL credentials from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/psql_credentials_readonly.py", "psql_credentials.py")
```

```python id="R5pArHA9BUJV"
# Import SQL credentials
from psql_credentials import PGUSER, PGPASSWORD, PGHOST, PGPORT, PGDATABASE

# Create the database connection string
connection_uri = f'postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}'
engine = create_engine(connection_uri)
```

<!-- #region id="uNJuPTnlBf3C" -->
## Fetch the ROOT file corresponding to the neutron burst
<!-- #endregion -->

<!-- #region id="BXAl9Bq4BMNs" -->
We must specify a channel number for where we neutron burst has been seen, the time of the burst and its duration.
<!-- #endregion -->

```python id="9oSGSyO0A_QD"
channel_number = 3
burst_time = '2025-06-16 16:19:45'
burst_duration = 60
```

<!-- #region id="PX5eyO2WJg8T" -->
We now look in the database for the closest ROOT file to the `burst_time`. Since our ROOT files are organised by their end time (time of the last event), we need to look for the closest ROOT file after our specified `burst_time`.
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

```python colab={"base_uri": "https://localhost:8080/", "height": 112} id="MULgCJIACxWN" outputId="cdf18730-fc02-49bd-84d3-5e0e1c676a58"
df_root = find_root_file(burst_time, channel_number)
df_root
```

<!-- #region id="2QLmFKCFN7UM" -->
Let's create a dictionary for easy access to the file info:
<!-- #endregion -->

```python id="nqjenrX_PLns"
closest_root_file = df_root.iloc[0].to_dict()
```

<!-- #region id="capsaKe7JnX-" -->
Now we need to construct the Google Drive path for this ROOT file to get its ID so that we can download it for use with uproot.
<!-- #endregion -->

```python id="nqUBfVWiJTg4"
def construct_file_path(root_file):
  return f"/content/drive/MyDrive/Computers/{root_file['computer']}/{root_file['dir']}/{root_file['file']}"

# Function to get file ID from file path
def get_file_id(file_path):
    query = f"name = '{file_path.split('/')[-1]}' and trashed = false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    if items:
        return items[0]['id']
    return None

def download_file_from_drive(file_id, filename):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(filename, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")

```

```python id="O0d0eZmvJUgK"
file_path = construct_file_path(closest_root_file)
file_id = get_file_id(file_path)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EgO6zwoTMlzN" outputId="bb3d6569-b045-4d55-ffbe-7e94f2bcb8f9"
filename = closest_root_file["file"]
download_file_from_drive(file_id, filename)
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
    ts = tree['Timestamp'].array(library='np')
    e = tree['Energy'].array(library='np')
    es = tree['EnergyShort'].array(library='np')
    wf = tree['Samples'].array(library='np')

# Creates dataframe with absolute time index
end_time = df_root.index[0]
start_time = end_time - pd.to_timedelta((ts[-1]-ts[0])/1e12, unit='s')
time_abs = start_time + pd.to_timedelta((ts - ts[0])/1e12, unit='s')
df_pulses = pd.DataFrame({'Energy': e, 'EnergyShort': es, 'Waveform': wf}, index=pd.to_datetime(time_abs))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="k-Q12sRwptwP" outputId="72c1f8b2-f166-4e42-d899-66d34629aac0"
df_pulses.head()
```

<!-- #region id="WbmLZnb7p2LP" -->
### Extracting only neutron pulses

Currently the `df_pulses` datafarme contains gammas and neutrons. The most accurate way to discriminate between gammas and neutrons is to create a fiducial lines. For the purpose of inspecting the waveforms to see whether a burst is physical of just electrical noise, a simple threshold PSP value is good enough. More discussion of this can be found in the [PSD Analysis notebook](https://github.com/project-ida/arpa-e-experiments/blob/main/tutorials/PSD-analysis.ipynb).

PSP is calculated by:

$$
\rm PSP = \frac{\rm Energy - EnergyShort}{\rm Energy}
$$

A neutron pulse is then defined as a pulse whose PSP is greater than a certain threshold.

<!-- #endregion -->

```python id="bTRM1Zt3CbKz"
psp_threshold = 0.15
```

```python id="pYmh_1b3lYmQ"
# Filtering out gammas to retain only neutrons
df_pulses['PSP'] = 1 - (df_pulses['EnergyShort'] / df_pulses['Energy'])

if psp_threshold:
  neutrons = df_pulses[df_pulses['PSP'] > psp_threshold]
else:
  print("‼️ PSP threshold not found ‼️ ")
```

<!-- #region id="KlFWvHRFrxS7" -->
### Visualising the counts

Let's now do a sanity check and visualise the counts per second over the entire ROOT file. The plot below should look identical to what we see in our data panels.

We'll also mark the burst time:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_erVbBxS9Bje" outputId="589ae9b7-f730-44c6-adc8-a90b26d80f43"
burst_time = pd.to_datetime(burst_time)
print(burst_time)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="HEQExazK5W_U" outputId="a872954f-3265-4a29-85f5-c9064d9bc920"
if psp_threshold:
  cps = neutrons['PSP'].resample('1s').count()
  plt.figure(figsize=(12, 4))
  plt.plot(cps, drawstyle='steps-mid')
  plt.axvline(burst_time, color='r', linestyle='--', label="Burst time")
  plt.title('Full ROOT file')
  plt.xlabel('Time')
  plt.ylabel('Counts per second')
  plt.legend()
  plt.grid(True)
  plt.show()
else:
  print("‼️ PSP threshold not found ‼️ ")
```

<!-- #region id="NbZfwRbD5oBS" -->
Let's now zoom in on the burst. Recall that our burst is defined as a time period centred on:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MmL3bAZs4jTz" outputId="b06b8fec-6192-4cd6-f732-f2465f4ecb04"
print(burst_time)
```

<!-- #region id="F9RaTGIW4nXA" -->
with a duration of:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="U6Yl3U8K4sDE" outputId="84242c73-4648-444e-8201-e9a11a700e51"
burst_duration = pd.to_timedelta(burst_duration, unit='s')
print(burst_duration)
```

<!-- #region id="WAdMOtV85BqZ" -->
Once we determine the `burst_start` and `burst_end` times, we can easily pull out cps and waveforms for the burst.
<!-- #endregion -->

```python id="8xyr7GiE3bV9"
burst_start = burst_time - burst_duration/2
burst_end = burst_time + burst_duration/2
```

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="3XbaW-nClas1" outputId="90f5eb41-fd6d-461b-9884-b3a5051a0fad"
if psp_threshold:
  plt.figure(figsize=(12, 4))
  plt.plot(cps[burst_start:burst_end], drawstyle='steps-mid', color='orange')
  plt.axvline(burst_time, color='r', linestyle='--', label="Burst time")
  plt.title('Burst period')
  plt.xlabel('Time')
  plt.ylabel('Counts per second')
  plt.legend()
  plt.grid(True)
  plt.show()
else:
  print("‼️ PSP threshold not found ‼️ ")
```

<!-- #region id="Sy9aHwBy24Co" -->
### Visualising the waveforms

We can now easily extract the visualise the neutron waveforms for the burst using the `burst_start` and `burst_end` times.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 545} id="uSxZYs4blcjF" outputId="92e7fc0e-2a80-4c34-f823-175d250dde9e"
if psp_threshold:
  df_burst = neutrons[burst_start:burst_end]
  plt.figure(figsize=(10, 6))
  for wf in df_burst['Waveform']:
      plt.plot(wf, alpha=0.5)
  plt.title(f'Waveforms for neutron pulses between {burst_start} and {burst_end}')
  plt.grid(True)
  plt.show()
else:
  print("‼️ PSP threshold not found ‼️ ")
```

```python id="h7F_gl3C_7Ee"

```
