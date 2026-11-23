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

<!-- #region id="2a9ece45-8fed-412c-b192-4fb1145b4d1b" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/mastromatteo-03/Mastromatteo_3_Radiation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/mastromatteo-03/Mastromatteo_3_Radiation.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="msqsXp7mp85M" -->

‼️ **Prerequisites** ‼️:
- Access to the `Nucleonics` Google drive folder (it must also be added as a shortcut called "Nucleonics" in your own drive)
- Access to the nucleonics `.env` folder (where sensitive info lives)
<!-- #endregion -->

<!-- #region id="PleVgUtQp_Zc" -->
# Radiation pulse analysis

The analysis notebook relies on the ["Nuclear particle master"](https://docs.google.com/spreadsheets/d/1zgp8MplLXNAI1s7NAs7tCCVnv7N05AR4Q7x4hnE_PiA/edit?gid=0#gid=0) sheet to provide timestamps for different phases of an experiment in order to perform bin-independent radiation analysis. The different phases are:

- Set-up
- Background 1
- Experiment
- Background 2

The aim is to develop techniques to characterise the background and detect anomalies in a way that minimises numerical artifiacts.

<!-- #endregion -->

<!-- #region id="nLIVEAOxqQYw" -->
## Running this notebook

Go ahead and change the `experiment_id` and `channel_number` below and then run the whole notebook.

You will be asked a couple of time to authenticate with your Google account, but after that all the analysis will happen automatically.
<!-- #endregion -->

```python id="n_BC5wa_rq9S"
experiment_id = 12
channel_number = 2
```

```python id="47c3211138ac" colab={"base_uri": "https://localhost:8080/"} outputId="1429a187-6940-4ea5-854b-cd3a7ea3ca9d"
# Makes Libs accessible and runs notebook from same location regardless of whether colab or local
!pip install colocal -q
import colocal
root, branch, cwd = colocal.setup()
```

```python id="8ede2344fa88"
# Fixes plotly install for proper rendering in nbviewer
!pip install -q --upgrade --force-reinstall "plotly==6.8.0"
```

<!-- #region id="gqHlPSyYrxDU" -->
## Libraries
<!-- #endregion -->

```python id="rnTZ6HBjrySX"
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
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sqlalchemy import create_engine, text
from scipy.stats import kstest


#Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# Make sure plotly displays properly in colab, notebooks and nbviewer
import plotly.io as pio
pio.renderers.default = "notebook_connected+colab"

# Use our custom helper functions
from libs.sql_api import SQLConnection, SQLAPIError, read_sql
```

<!-- #region id="3AaQ35teouCo" -->
## Authentication

We need to do a few authentication steps:
- Bring in the database credentials from Google drive so that we can pull data from the live database.
- Bring in the nuclear particle master sheet ID.
-  Authenticate Colab to pull the nuclear particle master sheet using the Drive API.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XK6IM7v-hnag" outputId="15c36385-a9ed-457c-e607-46feadad78ef"
# Mount Drive
drive.mount('/content/drive')

# Copy SQL credentials from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/psql_credentials_readonly.py", "psql_credentials.py")

# Copy sheet ID file from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/sheet_ids.py", "sheet_ids.py");
```

```python id="OA2IJWNSoyIF"
# Import SQL credentials
from psql_credentials import PGHOST, PGUSER, PGPASSWORD, PGPORT, PGDATABASE

# Import sheet ID for the nuclear particle master sheet
from sheet_ids import NUCLEAR_PARTICLE_MASTER as sheet_id

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

```python id="7xEZcdqnr8iF"
# Authenticate using Colab's built-in credentials
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
```

<!-- #region id="wP6CAlYos8R5" -->
## Extracting experimental timestamps

We need to
- Open the master sheet
- Find the row corresponding with the experiment
- Extract the timestamp columns

Because the master sheet is organised in blocks that share an experiment ID (to avoid visual overload) we'll need to "fill in" the experiment ID for all rows once the sheet is brought into a pandas dataframe.
<!-- #endregion -->

```python id="awvSNY0p0dXk"
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

```python id="IOwzthb8s-Mz"
sheet = gc.open_by_key(sheet_id).sheet1

# Read the sheet into a pandas DataFrame
df = pd.DataFrame(sheet.get_all_records())

# Fill the experiment IDs based on the block organisation of the master sheet
df = fill_experiment_id(df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="3Umn0vQM0nzf" outputId="2b1bdfb6-2438-40cb-a350-6d8c96f196c1"
# Find the rows where Experiment ID matches
rows = df[df['Experiment ID'] == experiment_id]

if len(rows) == 0:
  raise ValueError(f"No master sheet entry for 'Experiment ID' = {experiment_id}")

# Exract digitizer for SQL table identification
digitizer = rows.iloc[0]["Digitizer"]

# Extract the row corresponding to the specified channel number
row = rows[rows["Digitizer channel number"] == channel_number]

# Extract times
times = row.iloc[[0]][['Setup', 'Background 1', 'Experiment', 'Background 2', 'End']]
times = times.apply(pd.to_datetime)

if len(row) == 0:
  raise ValueError(f"No master sheet entry for 'Experiment ID' = {experiment_id}, 'Digitizer channel number' = {channel_number}")

# Extract the psp neutron/gamma discriminator
psp = row["psp threshold"].iloc[0]

# Display the extracted times
times.head()
```

<!-- #region id="UDmGFLkFtdsW" -->
## Pulling the radiation events

We store each individual radiation pulse in our database. These pulses are characterised by an energy and a psp value. The radiation pulses contain a mixture of gamma and neutron events which can be distinguised through PSD analysis at different levels of sophistication.

The [simplest PSD analysis](https://github.com/project-ida/arpa-e-experiments/blob/main/analysis/mastromatteo-03/Mastromatteo_3_PSD.ipynb) is to use a constant PSP discriminator value above which the pulses are considered to be neutrons, below are gammas. We can also combine this with an similar energy discriminator.

We can then query the database to pull only the events that match our PSD analysis requirements.
<!-- #endregion -->

```python id="JBq3UOFO5R5U"
def get_event_data(start_time, end_time, psp=">0", energy=">0"):
  query = f"""
  SELECT * FROM {digitizer}_ch{channel_number}
  WHERE channels[1] {psp} AND channels[1] < 1 AND channels[2] {energy}
  AND time BETWEEN '{start_time}' AND '{end_time}'
  ORDER BY time;
  """
  df = read_sql(query, engine, index_col=None)
  df.set_index('time', inplace=True)
  return df
```

<!-- #region id="tc1gXIIV5_en" -->
We can now get all the event data for the different periods in the experiement.
<!-- #endregion -->

```python id="kOyHsrTcs-IY"
def get_all_event_data(times, psp=">0", energy=">0"):
  event_data = {}
  event_periods = {}
  columns = times.columns  # Include all columns, including 'Setup'

  for i in range(len(columns) - 1):  # Stop before the last column
      start_time = times.iloc[0, i]
      if pd.notna(start_time):
          # Find the next non-empty time
          end_time = None
          for j in range(i + 1, len(columns)):
              if pd.notna(times.iloc[0, j]):
                  end_time = times.iloc[0, j]
                  break
          # Only proceed if a valid end_time was found
          if end_time is not None:
              data = get_event_data(start_time, end_time, psp, energy)
              event_data[columns[i]] = data
              event_periods[columns[i]] = end_time - start_time

  return event_data, event_periods
```

<!-- #region id="GoTVQZcE-H1m" -->
## Neutron background analysis

We're going to look at the neutron events during the background phase of the experiment and perform an analysis to characterise the background. Often, analysis of this kind is performed by "binning" the data and looking at counts per second/minute/hour. While it is often more intuitive to view the radiation rates, in our analysis we will take a bin-indepedent view by analysing the time in between neutron events. This way, we avoid any artifacts of binning.

Firstly, we need to filter out the neutrons from the gammas and reconstruct the neutron pulses to nanosecond precicison.
<!-- #endregion -->

<!-- #region id="EBdSq9e36GZA" -->
We can extract only the neutron events by using the psp values stored in the master spreadsheet.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XW3JZibgWexj" outputId="a88594c4-3756-4754-9fdf-46336cb44a69"
psp
```

<!-- #region id="03VCbjnTc1I-" -->
We'll also set a minimum energy channel to futher filter out gammas.
<!-- #endregion -->

```python id="5-LUemMIdKIy"
#Set energy cutoff in channels
energy = 500
```

```python id="cqnvIHxhtdTt"
neutron_data, neutron_periods = get_all_event_data(times, f">{psp}", f">{energy}")
```

<!-- #region id="qYRio4htXHRL" -->
### Reconstructing the pulses
<!-- #endregion -->

<!-- #region id="sUGMTuAstdD_" -->
Let's see what the pulse data looks like for the background.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="Q3hFn5Z5tXyv" outputId="79c4a641-926a-4578-8465-36f5b2e59911"
neutron_data["Background 1"].head()
```

<!-- #region id="LA8vcZtpEPmN" -->
The `channels` column gives `[psp, energy]`.

The database timestamp is limited in precision to microseconds, so we store the full picosecond precision in the `ps` columm. We therefore need to combine the time in microseconds with the picosecond time piece. It turns out that pandas only does datetime to nanoseconds so we'll have to discard some precision.
<!-- #endregion -->

```python id="u4kWrobGHvlO"
def reconstruct_ns_pulses(df):
  # Floor the index to strip microseconds
  df.index = df.index.floor('s')

  # Convert picoseconds to nanoseconds
  df['ps_ns'] = df['ps'] // 1_000  # convert ps → ns

  # Create high-resolution timestamp
  df['timestamp'] = df.index + pd.to_timedelta(df['ps_ns'], unit='ns')

  # Set timestamp as index
  df.set_index('timestamp', inplace=True)

  # Create 'counts' column with value 1 for each row
  df['counts'] = 1

  # Drop all unneeded columns (including 'channels')
  df.drop(columns=['id', 'ps', 'ps_ns', 'channels'], inplace=True)

  # Sort by timestamp index
  df.sort_index(inplace=True)

  return df
```

<!-- #region id="VO5bb1mBWuM5" -->
We'll now reconstruct the nanosecond pulses for all the following experimental periods:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="k_MsVojSGcot" outputId="3ac969d6-ae11-4044-92bc-98d02b8c0cab"
neutron_data.keys()
```

```python id="GASgu0SsIVCF"
for key, value in neutron_data.items():
  neutron_data[key] = reconstruct_ns_pulses(value)
```

<!-- #region id="RvySypWgXW-i" -->
### Inter-pulse distribution

#### Background

During the background phase of the experiment, we expect to measure radiation pulses randomly over time - following a Poisson distribution. This distribution can be studied by looking at count rates - as we do in the [Eljen Detector Background Characterization](https://github.com/project-ida/arpa-e-experiments/blob/main/tutorials/Eljen-Background-Characterization.ipynb) notebook. Here, we will study the distribution by instead looking at the times between individual events.

For a Poisson process occuring at an average rate of $\lambda$, the probability that the time between events $\Delta t$ is less than some time $t$ is given by:

$$
P_{\rm Poisson}(\Delta t \le t) = 1 - e^{-\lambda t}
$$

We can estimate the $\lambda$ from the experimentally derived average inter-pulse time $\overline{\Delta t}$ via:

$$
\lambda = \frac{1}{\overline{\Delta t}}
$$
<!-- #endregion -->

<!-- #region id="t9VUkOQbhgK4" -->
The background period we're going to look at lasted for:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LqId7nk-g_ul" outputId="a807b6e2-dd0e-4412-be3b-b26518287ab8"
neutron_periods["Background 1"]
```

<!-- #region id="dSRU5vnJhkjW" -->
Now let's perform the bin-independent analysis.
<!-- #endregion -->

```python id="rIj9BCQuVKGO"
background = neutron_data["Background 1"]
```

```python id="vZycjb_7tXuX"
deltas = np.diff(background.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
delta_sorted = np.sort(deltas)
```

```python id="TXshCDTAeGGZ"
# Estimate lambda (rate in events per second)
lam = 1 / np.mean(delta_sorted)

# Calculate the theoretical distribution
P_poisson = 1 - np.exp(-lam * delta_sorted)
```

<!-- #region id="WESwz1fwvA1U" -->
The background rate is
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MCON7zausVRZ" outputId="542ba8d0-e737-425b-d33a-db046b6a4bcd"
lam
```

<!-- #region id="IM3KEyowveG7" -->
counts per second.
<!-- #endregion -->

<!-- #region id="PhLfIKuXejvs" -->
We've now got the theoretical inter-pulse distribution based on the assumption that the system indeed behaves like a Poisson process. How does the system really behave?

We need to calculate the cumulative distribution of events in order to compare with the Poisson probability distribution.

$$
P_{\rm exp}(\Delta t \le t) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\bigl(\Delta t_i \le t\bigr)
$$
<!-- #endregion -->

```python id="TGNE79JotXp7"
P_exp = np.arange(1, len(delta_sorted) + 1) / len(delta_sorted)
```

<!-- #region id="pkQOzV0wkp6J" -->
Let's see how the Poisson distribution compares to the experimental one.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="qlhkMPk9s-F_" outputId="5f5e34c2-4ac0-4071-f6b2-bd6c53e44a0f"
plt.figure(figsize=(8, 5))
plt.plot(delta_sorted, P_exp, label="Empirical")
plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.5f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
plt.title("Cumulative inter-pulse time probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

<!-- #region id="jt_FGND9ky3u" -->
Visually, the level of agreement is adequate. The experimental data is noisey due to a relatively small sample size. Later in the notebook, a two-sample [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) compares the measured experiment and background inter-pulse distributions.
<!-- #endregion -->

<!-- #region id="eJ5bN8PjCQga" -->
Although we're interested primarily in the inter-pulse distribution, it's still helpful to look at the counts per minute.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 607} id="Gew5__t_rqmB" outputId="1204caca-fe35-487f-c2fe-78f53c26e5bd"
background_cpm = background.resample("60s").size().rename("counts").to_frame()

plt.figure(figsize=(8, 6))
plt.plot(background_cpm.index, background_cpm["counts"], label="Counts per min")
plt.ylabel("Counts per min")
plt.xlabel("Time")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))  # ISO-like
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Use below instead for interactive zoomable plot. Note this will not show up in nbviewer or Github
# fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=800))
# fig.add_trace(go.Scattergl(name="Counts per min", x=background_cpm.index, y=background_cpm.counts))

```

<!-- #region id="H1b5txOTgrGQ" -->
#### Experiment

Now let's perform the same analysis on the experimental time period as we did for the background period. The experiment lased for:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fI1Px1R9hYag" outputId="90c99e32-f0a3-42a4-9848-e8e5222e06cc"
neutron_periods["Experiment"]
```

```python id="GZnKQbGIhbZo"
experiment = neutron_data["Experiment"]
experiment_deltas = np.diff(experiment.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
experiment_delta_sorted = np.sort(experiment_deltas)
```

```python id="ClL9p_vUhdcY"
# Estimate lambda for experiment period (rate in events per second)
experiment_lam = 1 / np.mean(experiment_delta_sorted)

# Calculate the theoretical distribution
experiment_P_poisson = 1 - np.exp(-experiment_lam * experiment_delta_sorted)

experiment_P_exp = np.arange(1, len(experiment_delta_sorted) + 1) / len(experiment_delta_sorted)
```

<!-- #region id="xp3CoueMvnB8" -->
The rate during the experiment is
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GOmOopG4he-4" outputId="d410bf4b-bfa8-4a65-d619-204b0f1b70ed"
experiment_lam
```

<!-- #region id="cOhGyZUkvn3l" -->
counts per second.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="9-uSx8z5hgt4" outputId="dd81a7e9-7216-4b6a-acb2-c5e131f7b3b2"
plt.figure(figsize=(8, 5))
plt.plot(experiment_delta_sorted, experiment_P_exp, label="Empirical")
plt.plot(experiment_delta_sorted, experiment_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {experiment_lam:.5f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
#plt.xlim([0,15])
plt.title("Cumulative inter-pulse time probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 607} id="YeGF9sdxsgUh" outputId="fd1b1e03-7795-443b-c3bc-13f27a2f84c9"

experiment_cpm = experiment.resample("60s").size().rename("counts").to_frame()

plt.figure(figsize=(12, 6))
plt.plot(experiment_cpm.index, experiment_cpm["counts"], label="Counts per min")
plt.ylabel("Counts per min")
plt.xlabel("Time")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Use below instead for interactive zoomable plot. Note this will not show up in nbviewer or Github
# fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=1200))
# fig.add_trace(go.Scattergl(name="Counts per min", x=experiment_cpm.index, y=experiment_cpm.counts))
```

```python id="pLSqG10Ol310" colab={"base_uri": "https://localhost:8080/", "height": 543} outputId="1fede647-5a56-4165-e35b-49f8916658a4"
# Two-sample KS comparison of measured background and experiment gaps
n = len(delta_sorted)
m = len(experiment_delta_sorted)
if n == 0 or m == 0:
    raise ValueError("Both background and experiment need inter-event times.")

# Significance level and large-sample two-sample critical distance
alpha = 0.003
D_alpha = np.sqrt(-0.5 * np.log(alpha / 2)) * np.sqrt((m + n) / (m * n))

# Evaluate both empirical CDFs at every observed gap, including their steps
ks_grid = np.unique(np.concatenate(([0.0], delta_sorted, experiment_delta_sorted)))
background_cdf = np.searchsorted(delta_sorted, ks_grid, side="right") / n
experiment_cdf = np.searchsorted(experiment_delta_sorted, ks_grid, side="right") / m

upper_bound = np.clip(background_cdf + D_alpha, 0, 1)
lower_bound = np.clip(background_cdf - D_alpha, 0, 1)

# Check the complete distributions, including values outside the plotted range
cdf_difference = np.abs(experiment_cdf - background_cdf)
ks_statistic = np.max(cdf_difference)
crossing_indices = np.flatnonzero(cdf_difference > D_alpha)
first_cross_index = int(crossing_indices[0]) if crossing_indices.size else None

print(f"KS distance: {ks_statistic:.5g}; approx. critical distance (α={alpha}): {D_alpha:.5g}")
print("Experiment crosses the background band." if first_cross_index is not None
      else "Experiment remains within the background band.")

plt.figure(figsize=(8, 5))
plt.step(ks_grid, background_cdf, where="post", label="Background")
plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.5f}/s)")
plt.step(ks_grid, experiment_cdf, where="post", label=f"Experiment (λ = {experiment_lam:.5f}/s)")
plt.fill_between(ks_grid, lower_bound, upper_bound, step="post", color='gray', alpha=0.3,
                 label="Approx. two-sample KS band (α = 0.003)", zorder=1)

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t)")
#plt.xlim([0,1])
#plt.ylim([0,0.6])
plt.title(f"Cumulative inter-pulse time probability. PSP > {psp}, Channel > {energy}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
