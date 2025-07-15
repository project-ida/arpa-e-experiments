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
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/tutorials/neutron-waveform-extraction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/tutorials/neutron-waveform-extraction.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="ydG__I6QBDGM" -->

‼️ **Prerequisites** ‼️:
- Access to the `Nucleonics` Google drive folder (it must also be added as a shortcut called "Nucleonics" in your own drive)
- Access to the nucleonics `.env` folder (where sensitive info lives)
<!-- #endregion -->

<!-- #region id="oG6NQb5BBF0p" -->
# Neutron waveform extraction

This notebook extracts and visualises neutron event waveforms from a ROOT file. It relies on the ["Nuclear particle master"](https://docs.google.com/spreadsheets/d/1zgp8MplLXNAI1s7NAs7tCCVnv7N05AR4Q7x4hnE_PiA/edit?gid=0#gid=0) sheet to provide a PSP threshold value to separate neutrons from gammas. More information about how this PSP value has been calculated can be found in the [PSD Analysis notebook](https://github.com/project-ida/arpa-e-experiments/blob/main/tutorials/PSD-analysis.ipynb).



<!-- #endregion -->

<!-- #region id="80lfcZzf_5h1" -->
## Running this notebook

Once you specify the four parameters below, you can run the whole notebook from top to bottom.

- `experiment_id` & `channel_number`: select the experiment metadata row from the master sheet
- `burst_time`: time of the observed neutron burst
- `burst_duration`: time window (in s) around the burst from which we extract the waveforms

You will be asked a couple of time to authenticate with your Google account, but after that all the analysis will happen automatically.
<!-- #endregion -->

```python id="AT7BcrrpB0TV"
experiment_id = 6
channel_number = 3
burst_time = '2025-06-16 16:19:45'
burst_duration = 60
```

<!-- #region id="Fu4x4NPbBKrT" -->
## Libraries
<!-- #endregion -->

```python id="IAyoM46dBFqg"
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

```python colab={"base_uri": "https://localhost:8080/"} id="FhhgoaA3A0-Z" outputId="0ff16d17-bce7-4b9f-b6e3-3f70dcd71f35"
# Mount Drive
drive.mount('/content/drive')

# Authenticate and create the Drive API client
auth.authenticate_user()
drive_service = build('drive', 'v3')
creds, _ = default()
gc = gspread.authorize(creds)

# Copy SQL credentials from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/psql_credentials.py", "psql_credentials.py")

# Copy sheet ID file from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/sheet_ids.py", "sheet_ids.py");
```

```python id="R5pArHA9BUJV"
# Import SQL credentials
from psql_credentials import PGUSER, PGPASSWORD, PGHOST, PGPORT, PGDATABASE

# Import sheet ID for the nuclear particle master sheet
from sheet_ids import NUCLEAR_PARTICLE_MASTER as sheet_id

# Create the database connection string
connection_uri = f'postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}'
engine = create_engine(connection_uri)
```

<!-- #region id="N_VEuvAbFnmT" -->
## Extracting the PSP threshold for neutron discrimination

We need to

- Open the master sheet
- Find the row corresponding with the experiment and channel number:
- Extract the `psp threshold` column

Because the master sheet is organised in blocks that share an experiment ID (to avoid visual overload) we'll need to "fill in" the experiment ID for all rows once the sheet is brought into a pandas dataframe.
<!-- #endregion -->

```python id="OuSLjzoFKdGg"
# Fill experiment IDs based on the block organisation of the master sheet
def fill_experiment_id(df):
    experiment_id = None
    updated_ids = []

    for index, row in df.iterrows():
        if pd.notna(row['Experiment ID']) and row['Experiment ID'] != '':
            experiment_id = row['Experiment ID']
        updated_ids.append(experiment_id)

    df['Experiment ID'] = updated_ids
    return df
```

```python id="Vv4yexG-GESx"
sheet = gc.open_by_key(sheet_id).sheet1

# Read the sheet into a pandas DataFrame
df_sheet = pd.DataFrame(sheet.get_all_records())

# Fill the experiment IDs based on the block organisation of the master sheet
df_sheet = fill_experiment_id(df_sheet)
```

```python colab={"base_uri": "https://localhost:8080/"} id="rjajUnhCHBoO" outputId="448e822e-ef2a-4106-e779-9837fe18302e"
# Find the row where Experiment ID and channel number match
row = df_sheet[(df_sheet['Experiment ID'] == experiment_id) & (df_sheet['Digitizer channel number'] == channel_number)]

if(len(row) == 0):
  raise RuntimeError("‼️ No row found, check your Experiment ID and Digitizer channel number ‼️ ")

# Extract times
psp_threshold = row.iloc[0]["psp threshold"]

if psp_threshold:
  print(f"PSP threshold: {psp_threshold}")
else:
  print("‼️ PSP threshold not found ‼️ ")
```

<!-- #region id="uNJuPTnlBf3C" -->
## Fetch the ROOT file corresponding to the neutron burst
<!-- #endregion -->

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

```python colab={"base_uri": "https://localhost:8080/", "height": 112} id="MULgCJIACxWN" outputId="7542895c-413c-4134-9550-c81fa80f9430"
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

```python colab={"base_uri": "https://localhost:8080/"} id="EgO6zwoTMlzN" outputId="2dc3bedd-f58b-434a-9edd-318961d9b6c9"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="k-Q12sRwptwP" outputId="3d756100-ca39-4f54-9c4b-857a07753c94"
df_pulses.head()
```

<!-- #region id="WbmLZnb7p2LP" -->
### Extracting only neutron pulses

Currently the `df_pulses` datafarme contains gammas and neutrons. The most accurate way to discriminate between gammas and neutrons is to create a fiducial lines. For the purpose of inspecting the waveforms to see whether a burst is physical of just electrical noise, a simple threshold PSP value is good enough. More discussion of this can be found in the [PSD Analysis notebook](https://github.com/project-ida/arpa-e-experiments/blob/main/tutorials/PSD-analysis.ipynb).

PSP is calculated by:

$$
\rm PSP = \frac{\rm Energy - EnergyShort}{\rm Energy}
$$

A neutron pulse is then defined as a pulse whose PSP is greater than the threshold pulled in from the master sheet.

<!-- #endregion -->

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

```python colab={"base_uri": "https://localhost:8080/"} id="_erVbBxS9Bje" outputId="e14f56db-cc1f-4d65-f4f2-425cb3bbabfc"
burst_time = pd.to_datetime(burst_time)
print(burst_time)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="HEQExazK5W_U" outputId="87d49baa-4c44-47eb-f32b-eda139c9f0e2"
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

```python colab={"base_uri": "https://localhost:8080/"} id="MmL3bAZs4jTz" outputId="f5339420-6970-4f24-b060-3c101dba362c"
print(burst_time)
```

<!-- #region id="F9RaTGIW4nXA" -->
with a duration of:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="U6Yl3U8K4sDE" outputId="7d06a1cf-a958-4642-aa81-176b1e4192e5"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="3XbaW-nClas1" outputId="b7f6ad86-273d-4f1e-b600-0dc1d2741078"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 545} id="uSxZYs4blcjF" outputId="0cee63c4-1ee8-49f3-98a5-41442490462c"
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
