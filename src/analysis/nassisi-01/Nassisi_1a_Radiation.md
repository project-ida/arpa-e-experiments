---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.7
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="99b2add5-fe11-4c53-b380-977308ec2bbf" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-01/Nassisi_1a_Radiation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-01/Nassisi_1a_Radiation.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="msqsXp7mp85M" -->

‼️ **Prerequisites** ‼️:
- Access to the `Nucleonics` Google drive folder (it must also be added as a shortcut called "Nucleonics" in your own drive)
- Access to the nucleonics `.env` folder (where sensitive info lives)
<!-- #endregion -->

<!-- #region id="PleVgUtQp_Zc" -->
# Radiation pulse analysis

The analysis notebook relies on the "Nuclear particle master" sheet to provide timestamps for different phases of an experiment in order to perform bin-independent radiation analysis.

The aim is to develop techniques to characterise the background and detect anomalies in a way that minimises numerical artifiacts.

<!-- #endregion -->

<!-- #region id="nLIVEAOxqQYw" -->
## Nassisi 1a

The first Nassisi style experiment ran from April 4 to June 20, 2025 in chamber #1. A 9 cm Pd wire was coiled and placed in 3.22 bar absolute pressure of D2 gas for one month. Then, the chamber was evacuated and vented to atmosphere, before being resealed and repressurized with 2.34 bar absolute pressure of D2, followed by exposure to a UV pulsed laser for one hour a day for one month.

This notebook analyzes the neutron emission during the first half of this experiment as measured by a 2" Eljen detector placed just outside the chamber. Following the end of the soaking period, the chamber was transported to building 2 for laser irradiation. During transit, a cable was damaged, prompting a change from digitizer channel 0 to channel 1. The second half of this experiment, being recorded on a different channel, is analyzed in a separate notebook here: https://colab.research.google.com/drive/1DSHfJ7VeaKeE-AuhLKChBjujVEkutWg0?usp=sharing
<!-- #endregion -->

```python id="n_BC5wa_rq9S"
experiment_id = 3
```

```python colab={"base_uri": "https://localhost:8080/"} id="7a063e0cd5eb" outputId="f6ec87ff-97eb-45ee-f2a9-9af773149162"
# Makes Libs accessible and runs notebook from same location regardless of whether colab or local
!pip install colocal -q
import colocal
root, branch, cwd = colocal.setup()
```

```python id="0a135e881a52"
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
from scipy.signal import find_peaks
from sqlalchemy import create_engine, text
from scipy.stats import ks_2samp


#Plotting
import matplotlib.pyplot as plt
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
- Bring in the nuclear particle master sheet ID
-  Authenticate Colab to pull the nuclear particle master sheet using the Drive API.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XK6IM7v-hnag" outputId="afec19e0-39d3-4cd7-e7b0-d092d493b2ec"
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
<!-- #endregion -->

```python id="IOwzthb8s-Mz"
sheet = gc.open_by_key(sheet_id).sheet1

# Read the sheet into a pandas DataFrame
df = pd.DataFrame(sheet.get_all_records())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="RndMn9Zos-LC" outputId="665e37a1-d3ac-4f28-e721-f7a6879d2392"
# Find the row where Experiment ID matches
row = df[df['Experiment ID'] == experiment_id]

# Exract digitizer, either 4 channel or 8
digitizer = 8 #row["Digitizer"].iloc[0]

# Extract the channel number
channel_number = row["Digitizer channel number"].iloc[0]

# Extract the psp neutron/gamma discriminator
psp = row["psp threshold"].iloc[0]

# Extract times
times = row[['Setup', 'Calibration', 'Background 1', 'Experiment', 'Background 2', 'End']]

times = times.apply(pd.to_datetime)

# Display the extracted times
times.head()
```

<!-- #region id="UDmGFLkFtdsW" -->
## Pulling the radiation events

We store each individual radiation pulse in our database. These pulses are characterised by an energy and a psp value. The radiation pulses contain a mixture of gamma and neutron events which can be distinguised through PSD analysis at different levels of sophistication.

The [simplest PSD analysis](https://github.com/project-ida/arpa-e-experiments/blob/main/tutorials/PSD-analysis.ipynb) is to use a PSP discriminator value above which the pulses are considered to be neutrons, below are gammas. We can also combine this with an similar energy discriminator.

We can then query the database to pull only the events that match our PSD analysis requirements.
<!-- #endregion -->

```python id="JBq3UOFO5R5U"
def get_event_data(start_time, end_time, psp=">0", energy=">0"):
  query = f"""
  SELECT * FROM caen{digitizer}ch_ch{channel_number}
  WHERE channels[1] {psp} AND channels[2] {energy}
  AND time BETWEEN '{start_time}' AND '{end_time}'
  ORDER BY time;
  """
  df = read_sql(query, engine, index_col=None)
  df.set_index('time', inplace=True)
  return df
```

<!-- #region id="tc1gXIIV5_en" -->
We'll can now get all the event data for the different periods in the experiement.
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
<!-- #endregion -->

<!-- #region id="EBdSq9e36GZA" -->
We can extract only the neutron events by using the psp values stored in the master spreadsheet.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XW3JZibgWexj" outputId="5c8fc6fc-e901-491c-f12b-ea6e32c19be2"
psp
```

```python id="WgF4FZvAJO-D"
#Set energy cutoff in channel
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

```python colab={"base_uri": "https://localhost:8080/", "height": 175} id="Q3hFn5Z5tXyv" outputId="e1783bf8-7d48-489f-d373-5351314b4dee"
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
We'll now reconstruct the nanosecond pulses for all the experimental periods.
<!-- #endregion -->

```python id="GASgu0SsIVCF"
for key, value in neutron_data.items():
  neutron_data[key] = reconstruct_ns_pulses(value)
```

<!-- #region id="RvySypWgXW-i" -->
### Inter-pulse distribution

During the background phase of the experiment, we expect to measure radiation pulses randomly over time - following a Poisson distribution.

For a Poisson process occuring at an average rate of $\lambda$, the probability that the time between events $\Delta t$ is less than some time $t$ is given by:

$$
P_{\rm Poisson}(\Delta t \le t) = 1 - e^{-\lambda t}
$$

We can estimate the $\lambda$ from the experimentally derived average inter-pulse time $\overline{\Delta t}$ via:

$$
\lambda = \frac{1}{\overline{\Delta t}}
$$
<!-- #endregion -->

```python id="rIj9BCQuVKGO"
background = neutron_data["Background 1"]
```

```python id="vZycjb_7tXuX"
deltas = np.diff(background.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
delta_sorted = np.sort(deltas)
```

```python colab={"base_uri": "https://localhost:8080/"} id="TXshCDTAeGGZ" outputId="ed2c2f90-8a06-47c0-c8e4-e266456f1d60"
# Estimate lambda (rate in events per second)
lam = 1 / np.mean(delta_sorted)

# Calculate the theoretical distribution
P_poisson = 1 - np.exp(-lam * delta_sorted)
lam
```

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

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="qlhkMPk9s-F_" outputId="5b1ecd44-215e-42ab-f811-d4e6aad77190"
plt.figure(figsize=(8, 5))
plt.plot(delta_sorted, P_exp, label="Empirical")
plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.4f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
plt.title("Background 1 cumulative inter-pulse time probability")
plt.xlim(left=0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

<!-- #region id="jt_FGND9ky3u" -->
The background follows a Poisson distribution, with some noise following from the shorter time period of background 1 (40 minutes).

This background was recorded in 13-3100.
<!-- #endregion -->

<!-- #region id="eJ5bN8PjCQga" -->
It's instructive to look at the cumulative pulses alongside the counts per minute.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="iPSuT_3dJ5kp" outputId="054665ac-6443-46db-e37e-999da7c7aedd"
background_cpm = background.resample("60s").size().rename("counts").to_frame()
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=800))
fig.add_trace(go.Scattergl(name="Counts per min", x=background_cpm.index, y=background_cpm.counts))
```

<!-- #region id="B8ZzYbhLlMNV" -->
## Anomaly detection

We can use the inter-pulse cumulative probability to detect deviations from normal background, aka anomalies.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PdHlw300IfEy" outputId="d0f84e36-f806-4d95-ce06-b5676770b4bf"
neutron_periods["Experiment"]

```

<!-- #region id="S-J0ASGFACkz" -->
Because this experiment was switched from digitizer channel 0 to digitizer channel 1 (due to a damaged cable during transportation from 13-3100 to 2-073), this data set only includes the soaking period which took place in 13-3100.
<!-- #endregion -->

```python id="3caxZrbEpInb"
experiment = neutron_data["Experiment"]
experiment_deltas = np.diff(experiment.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
#Exclude deltas greater than 6 hours to avoid gaps in the data acquisition
experiment_deltas = experiment_deltas[experiment_deltas < 21600]
experiment_delta_sorted = np.sort(experiment_deltas)
```

```python id="8YicCCICHk3W"
# Estimate lambda for experiment period (rate in events per second)
experiment_lam = 1 / np.mean(experiment_delta_sorted)

# Calculate the theoretical distribution
experiment_P_poisson = 1 - np.exp(-experiment_lam * experiment_delta_sorted)

experiment_P_exp = np.arange(1, len(experiment_delta_sorted) + 1) / len(experiment_delta_sorted)
```

```python colab={"base_uri": "https://localhost:8080/"} id="R5l7VjXUAJCx" outputId="a5a10bba-ee72-4754-85e3-d853759ae03a"
experiment_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="hXgGNo7aIHsn" outputId="9b21076d-885f-4f65-b13f-72734cd38b1e"
plt.figure(figsize=(8, 5))
plt.plot(experiment_delta_sorted, experiment_P_exp, label="Empirical")
plt.plot(experiment_delta_sorted, experiment_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {experiment_lam:.4f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
plt.xlim([0,60])
plt.title("Cumulative inter-pulse time probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<!-- #region id="5Vbmd1_tA5dC" -->
The data for this period also follows a Poisson distribution, suggesting that the neutron events are not correlated in time. It is notable that the rate is higher than during the background period; however, there are 3 spikes in the count rate which are associated with electronic issues (disconnecting and reconnecting):
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="JdyP7cveiSOR" outputId="9f1f568e-debe-4d6a-ccdf-17f057923610"
experiment_cpm = experiment.resample("60s").size().rename("counts").to_frame()
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=1200))
fig.add_trace(go.Scattergl(name="Counts per min", x=experiment_cpm.index, y=experiment_cpm.counts))
```

<!-- #region id="JZbwIIHfBR_q" -->
The following timing analysis includes the electronic spikes. We then repeat the analysis with an earlier end time to exclude the final period affected by these disturbances.
<!-- #endregion -->

<!-- #region id="6McultY7klTK" -->
We now compare the soaking-period inter-event times with the measured Background 1 sample. The two-sample KS band uses both sample sizes and is centred on the background cumulative distribution. Crossing the band corresponds to rejection by the exact finite-sample two-sample KS test at a significance level of 0.003.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="T57-XEeqCaeR" outputId="43e1b9cf-8dca-4fff-d1c2-b232a3b2bb00"
# Compare the measured background and experiment inter-event times
ks_stat, p_value = ks_2samp(
    delta_sorted, experiment_delta_sorted, alternative="two-sided", method="exact"
)

ks_stat, p_value
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="juDzyr-9Cz9r" outputId="199d7ccb-5155-4189-fcaf-33c82a8f74b3"
def exact_two_sample_ks_band(n, m, alpha=0.003):
    """Return a band width matching the exact two-sided KS rejection rule."""
    from fractions import Fraction
    from math import comb, gcd

    if n < 1 or m < 1:
        raise ValueError("Both samples must contain inter-event intervals.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1.")

    n, m = sorted((n, m))
    g = gcd(n, m)
    lattice_size = n * m // g
    total_paths = comb(n + m, n)
    level = Fraction(str(alpha))

    def tail_count(k):
        # Count pooled-rank paths with D < k / lattice_size.
        paths = [0] * (n + 1)
        paths[0] = 1
        for j in range(m + 1):
            for i in range(n + 1):
                if abs(i * m - j * n) >= k * g:
                    paths[i] = 0
                elif i:
                    paths[i] += paths[i - 1]
        return total_paths - paths[n]

    # Find the first distance with exact tail probability <= alpha.
    lo, hi = 0, lattice_size + 1
    while hi - lo > 1:
        k = (lo + hi) // 2
        if tail_count(k) * level.denominator <= total_paths * level.numerator:
            hi = k
        else:
            lo = k

    # A midpoint between attainable distances makes a strict band crossing
    # equivalent to p <= alpha. If rejection is impossible, use the full range.
    return min(1.0, (hi - 0.5) / lattice_size)


# Number of measured background (n) and experiment (m) inter-event times
n = len(delta_sorted)
m = len(experiment_delta_sorted)
if n == 0 or m == 0:
    raise ValueError("Both periods need inter-event times for the two-sample KS comparison.")

# Significance level and exact finite-sample two-sample KS band width
alpha = 0.003
D_alpha = exact_two_sample_ks_band(n, m, alpha)

# Evaluate both empirical CDFs at every observed inter-event time
ks_times = np.unique(np.concatenate(([0.0], delta_sorted, experiment_delta_sorted)))
background_cdf = np.searchsorted(delta_sorted, ks_times, side="right") / n
experiment_cdf = np.searchsorted(experiment_delta_sorted, ks_times, side="right") / m

# Centre the comparison band on the measured background
upper_bound = np.clip(background_cdf + D_alpha, 0, 1)
lower_bound = np.clip(background_cdf - D_alpha, 0, 1)

# Locate crossings using the same two-sample comparison
crossing_indices = np.flatnonzero(np.abs(experiment_cdf - background_cdf) > D_alpha)
first_cross_index = int(crossing_indices[0]) if crossing_indices.size else None

plt.figure(figsize=(8, 5))
plt.step(ks_times, background_cdf, where="post", label="Background 1")
plt.step(ks_times, experiment_cdf, where="post", label="Experiment")
plt.plot(experiment_delta_sorted, experiment_P_poisson, linestyle="--", color="red", label=f"Experiment fit (λ = {experiment_lam:.4f}/s)")
plt.fill_between(ks_times, lower_bound, upper_bound, step="post", color='gray', alpha=0.3, label="Exact two-sample KS band (α = 0.003)", zorder=1)

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t)")
#plt.xlim([0,5])
#plt.ylim([0,0.2])
plt.title(f"Soaking Period Cumulative inter-pulse time probability. PSP > {psp}, Channel > {energy}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<!-- #region id="Wk6mo9ask3cr" -->
The electronic spikes contain roughly 5000 data points with artificially low time separations. We next drop the final day to exclude these spikes and repeat the comparison with Background 1.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="k2m2Fws3OjOz" outputId="8cee8bbf-430e-44ef-85b0-f202352a9c73"
times['End'] = "2025-05-07 11:32:00"
times = times.apply(pd.to_datetime)
times.head()
```

```python id="Xiw3YcsJO8cT"
neutron_data, neutron_periods = get_all_event_data(times, f">{psp}", f">{energy}")
```

```python id="bhn6p1hPPMny"
for key, value in neutron_data.items():
  neutron_data[key] = reconstruct_ns_pulses(value)
```

```python colab={"base_uri": "https://localhost:8080/"} id="hP3QlodTPNfE" outputId="06ca369a-e400-4587-dfed-86f1c0be4cb0"
neutron_periods["Experiment"]
```

```python id="KuBI5gkSPTJC"
experiment = neutron_data["Experiment"]
experiment_deltas = np.diff(experiment.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
#Exclude deltas greater than 6 hours to avoid gaps in the data acquisition
experiment_deltas = experiment_deltas[experiment_deltas < 21600]
experiment_delta_sorted = np.sort(experiment_deltas)
```

```python id="RSrW5eU6PXAS"
# Estimate lambda for experiment period (rate in events per second)
experiment_lam = 1 / np.mean(experiment_delta_sorted)

# Calculate the theoretical distribution
experiment_P_poisson = 1 - np.exp(-experiment_lam * experiment_delta_sorted)

experiment_P_exp = np.arange(1, len(experiment_delta_sorted) + 1) / len(experiment_delta_sorted)
```

```python colab={"base_uri": "https://localhost:8080/"} id="B5RDtAu2PasZ" outputId="02126a59-f95a-46a7-f5c5-3a5915f05fcb"
experiment_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="1kWumQ9GPjUU" outputId="0d3ad9e3-7b29-4fb4-8ac5-050069a37025"
# Number of measured background (n) and experiment (m) inter-event times
n = len(delta_sorted)
m = len(experiment_delta_sorted)
if n == 0 or m == 0:
    raise ValueError("Both periods need inter-event times for the two-sample KS comparison.")

# Significance level and exact finite-sample two-sample KS band width
alpha = 0.003
D_alpha = exact_two_sample_ks_band(n, m, alpha)

# Evaluate both empirical CDFs at every observed inter-event time
ks_times = np.unique(np.concatenate(([0.0], delta_sorted, experiment_delta_sorted)))
background_cdf = np.searchsorted(delta_sorted, ks_times, side="right") / n
experiment_cdf = np.searchsorted(experiment_delta_sorted, ks_times, side="right") / m

# Centre the comparison band on the measured background
upper_bound = np.clip(background_cdf + D_alpha, 0, 1)
lower_bound = np.clip(background_cdf - D_alpha, 0, 1)

# Locate crossings using the same two-sample comparison
crossing_indices = np.flatnonzero(np.abs(experiment_cdf - background_cdf) > D_alpha)
first_cross_index = int(crossing_indices[0]) if crossing_indices.size else None

plt.figure(figsize=(8, 5))
plt.step(ks_times, background_cdf, where="post", label="Background 1")
plt.step(ks_times, experiment_cdf, where="post", label="Experiment")
plt.plot(experiment_delta_sorted, experiment_P_poisson, linestyle="--", color="red", label=f"Experiment fit (λ = {experiment_lam:.4f}/s)")
plt.fill_between(ks_times, lower_bound, upper_bound, step="post", color='gray', alpha=0.3, label="Exact two-sample KS band (α = 0.003)", zorder=1)

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t)")
#plt.xlim([0,5])
#plt.ylim([0,0.2])
plt.title(f"Soaking Period Cumulative inter-pulse time probability. PSP > {psp}, Channel > {energy}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<!-- #region id="HHXYeLVsmM0e" -->
### Laser period
The preceding analysis was limited to the soaking period. The laser period was recorded on channel 1 on the digitizer. The laser period analyses can be found in https://colab.research.google.com/drive/1DSHfJ7VeaKeE-AuhLKChBjujVEkutWg0?authuser=2#scrollTo=_dpYHvPdhkb4
<!-- #endregion -->
