---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
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
# Waveform extraction

This notebook contains starter code that extracts and visualises event waveforms from a ROOT file.
<!-- #endregion -->

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

```python colab={"base_uri": "https://localhost:8080/", "height": 53} id="FhhgoaA3A0-Z" outputId="24e35c68-72bf-4e62-d92c-e4e3d47b84f7"
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
    df = pd.read_sql(query, con=engine, params={
        "event_time": event_time,
        "file_pattern": file_pattern
    })
    df.set_index('time', inplace=True)
    return df

```

```python colab={"base_uri": "https://localhost:8080/", "height": 112} id="MULgCJIACxWN" outputId="7f3c1dc2-788c-4e80-ad23-fee62d2f87c3"
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

```python colab={"base_uri": "https://localhost:8080/"} id="EgO6zwoTMlzN" outputId="42770c21-9fa1-4966-fc3d-ec73c6ae188e"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="k-Q12sRwptwP" outputId="d66633aa-3872-4839-f345-b20dc311f299"
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
df_pulses['PSP'] = 1 - (df_pulses['EnergyShort'] / df_pulses['Energy'])
```

```python id="bTRM1Zt3CbKz"
psp_range = [0,1] # All PSP values

energy_range = [0,70] # low energy
# energy_range = [71,600] # high energy
# energy_range = [901,3100] # very high energy
```

```python id="pYmh_1b3lYmQ"
filtered = df_pulses[(df_pulses['PSP'] > psp_range[0]) & (df_pulses['PSP'] < psp_range[1])]

filtered = filtered[(filtered['Energy'] > energy_range[0]) & (filtered['Energy'] < energy_range[1])]

```

<!-- #region id="KlFWvHRFrxS7" -->
### Visualising the counts

Let's now do a sanity check and visualise the counts per second over the entire ROOT file. The plot below should look identical to what we see in our data panels.

We'll also mark the burst time:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_erVbBxS9Bje" outputId="5c89cd55-9a99-4e1d-aeca-dd2867fa3c11"
time = pd.to_datetime(time)
print(time)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="HEQExazK5W_U" outputId="72edbc6e-a784-40b4-92ed-2d8543353f0e"
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

```python colab={"base_uri": "https://localhost:8080/"} id="MmL3bAZs4jTz" outputId="f5c3ea05-8cb3-4cb0-94ad-9fb6d373f548"
print(time)
```

<!-- #region id="F9RaTGIW4nXA" -->
with a duration of:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="U6Yl3U8K4sDE" outputId="b2552404-0b18-4672-a023-3fc5b5b76a44"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="3XbaW-nClas1" outputId="728ce366-b6ad-48cb-bd92-1fd8539bcecb"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 606} id="uSxZYs4blcjF" outputId="4ab6f704-ae08-4079-fb69-3700025452a3"
df_period = filtered[start:end]

plt.figure(figsize=(10, 6))
for wf in df_period['Waveform']:
    plt.plot(wf, alpha=0.5)
plt.title(f'Waveforms for pulses between {start} and {end} \n with PSP in range {psp_range}\n and energy in range {energy_range}')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

```python id="h7F_gl3C_7Ee"

```
