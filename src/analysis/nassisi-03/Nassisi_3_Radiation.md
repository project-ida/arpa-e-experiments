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

<!-- #region id="4889e770-8998-40ec-b48f-a8dc984323c4" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-03/Nassisi_3_Radiation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-03/Nassisi_3_Radiation.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
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
### Nassisi 3
The third Nassisi style experiment ran from April 4 to June 20, 2025 in chamber #3. A 15.4 cm Pd wire was coiled and placed in 3.20 bar absolute pressure of D2 gas for one month. Then, the chamber was evacuated and vented to atmosphere, before being resealed and repressurized with 2.34 bar absolute pressure of D2, followed by exposure to a UV pulsed laser for one hour a day for one month.

This notebook analyzes the neutron emission during this time period as measured by a 2" Eljen detector placed just outside the chamber.
<!-- #endregion -->

```python id="n_BC5wa_rq9S"
experiment_id = 6
```

```python id="a6e41008a89d" colab={"base_uri": "https://localhost:8080/"} outputId="e06f5583-f482-4b83-979f-f65230c4af5a"
# Makes Libs accessible and runs notebook from same location regardless of whether colab or local
!pip install colocal -q
import colocal
root, branch, cwd = colocal.setup()
```

```python id="a39657cfcea7"
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
from scipy.stats import kstest, ks_2samp


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

```python colab={"base_uri": "https://localhost:8080/"} id="XK6IM7v-hnag" outputId="b5676de2-6a2b-4bfd-fb1f-bf57fc184f64"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="RndMn9Zos-LC" outputId="e594d907-75e5-41d6-c581-0827e389ab12"
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

The [simplest PSD analysis](https://github.com/project-ida/arpa-e-experiments/blob/main/tutorials/PSD_Analysis.ipynb) is to use a PSP discriminator value above which the pulses are considered to be neutrons, below are gammas. We can also combine this with an similar energy discriminator.

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

```python colab={"base_uri": "https://localhost:8080/"} id="XW3JZibgWexj" outputId="fb1ac7bf-9a20-47f5-9928-84fc0b91f4ab"
psp
```

```python id="FcoBc7xEgFFp"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="Q3hFn5Z5tXyv" outputId="5fcae015-094e-4437-8f54-0da4d75ef091"
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

We can estimate the $\lambda$ from the experimetnally derived average inter-pulse time $\overline{\Delta t}$ via:

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

```python id="TXshCDTAeGGZ"
# Estimate lambda (rate in events per second)
lam = 1 / np.mean(delta_sorted)

# Calculate the theoretical distribution
P_poisson = 1 - np.exp(-lam * delta_sorted)
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

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="qlhkMPk9s-F_" outputId="1cdd382d-1dc1-44ed-8a72-7ea5fd0cedc2"
plt.figure(figsize=(8, 5))
plt.plot(delta_sorted, P_exp, label="Empirical")
plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.5f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
plt.title("Background cumulative inter-pulse time probability")
#plt.xlim([0,30])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

<!-- #region id="jt_FGND9ky3u" -->
**TODO: Statistical test**
<!-- #endregion -->

<!-- #region id="KinoI11EhsYG" -->
Let's also check the second background period, which was also taken in 13-3100, but for several days.
<!-- #endregion -->

```python id="jgKPPX30h0zu"
background2 = neutron_data["Background 2"]
```

```python id="yN0Oo3sKh2IG"
background2_deltas = np.diff(background2.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
background2_delta_sorted = np.sort(background2_deltas)
```

```python id="O6A03qO4iC2G"
# Estimate lambda (rate in events per second)
background2_lam = 1 / np.mean(background2_delta_sorted)

# Calculate the theoretical distribution
background2_P_poisson = 1 - np.exp(-background2_lam * background2_delta_sorted)

background2_P_exp = np.arange(1, len(background2_delta_sorted) + 1) / len(background2_delta_sorted)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ssCwTHVKiWEg" outputId="090e0465-3cb2-412e-d0b9-e92e705305f5"
background2_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="OEVPdqQEibrg" outputId="a1fa36d1-71cc-4b8c-f3df-1d7792c28cb2"
plt.figure(figsize=(8, 5))
plt.plot(background2_delta_sorted, background2_P_exp, label="Empirical")
plt.plot(background2_delta_sorted, background2_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {background2_lam:.5f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
plt.title("Background 2 cumulative inter-pulse time probability")
plt.xlim([0,30])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

<!-- #region id="vZ5l_hCiTb6a" -->
# Run KS test against an exponential distribution with estimated lambda
ks_stat, p_value = kstest(synthetic_deltas_sorted, 'expon', args=(0, 1/lam))

ks_stat, p_value
<!-- #endregion -->

```python id="x-xLJnC9Tbf-"
# # Run KS test against an exponential distribution with estimated lambda
# ks_stat, p_value = kstest(synthetic_deltas_sorted, 'expon', args=(0, 1/lam))

# ks_stat, p_value
```

<!-- #region id="eJ5bN8PjCQga" -->
It's instructive to look at the cumulative pulses alongside the counts per minute.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="iPSuT_3dJ5kp" outputId="353a1a57-a13a-4f74-c073-8fde8430d072"
background_cpm = background.resample("60s").size().rename("counts").to_frame()
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=800))
fig.add_trace(go.Scattergl(name="Counts per min", x=background_cpm.index, y=background_cpm.counts))
```

<!-- #region id="B8ZzYbhLlMNV" -->
## Anomaly detection

We can use the inter-pulse cumulative probability to detect deviations from normal background, aka anomalies.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PdHlw300IfEy" outputId="db4d718c-0aa0-4ba0-c689-4af934e23d1f"
neutron_periods["Experiment"]
```

```python id="3caxZrbEpInb"
experiment = neutron_data["Experiment"]
experiment_deltas = np.diff(experiment.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
experiment_delta_sorted = np.sort(experiment_deltas)
```

```python id="8YicCCICHk3W"
# Estimate lambda for experiment period (rate in events per second)
experiment_lam = 1 / np.mean(experiment_delta_sorted)

# Calculate the theoretical distribution
experiment_P_poisson = 1 - np.exp(-experiment_lam * experiment_delta_sorted)

experiment_P_exp = np.arange(1, len(experiment_delta_sorted) + 1) / len(experiment_delta_sorted)
```

```python colab={"base_uri": "https://localhost:8080/"} id="R5l7VjXUAJCx" outputId="70a46525-5cfc-46ec-fa24-fc39df91fc3c"
experiment_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="hXgGNo7aIHsn" outputId="a6ff829c-dd08-468e-e6e0-cb7e9af25b81"
plt.figure(figsize=(8, 5))
plt.plot(experiment_delta_sorted, experiment_P_exp, label="Empirical")
plt.plot(experiment_delta_sorted, experiment_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {experiment_lam:.5f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
#plt.xlim([0,30])
plt.title("Cumulative inter-pulse time probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="M1BAZ4z5pPeL" outputId="b496e86a-7674-43d3-afcd-7f61c5596e3c"
neutron_periods["Background 1"]
```

```python id="T57-XEeqCaeR" colab={"base_uri": "https://localhost:8080/"} outputId="e3e16a08-804c-4df3-e74d-ecf0d7a3bf9a"
# Two-sample KS test of measured experiment and Background 2 inter-event times
ks_stat, p_value = ks_2samp(
    experiment_delta_sorted, background2_delta_sorted,
    alternative="two-sided", method="auto",
)

ks_stat, p_value
```

```python id="juDzyr-9Cz9r" colab={"base_uri": "https://localhost:8080/", "height": 507} outputId="30adc238-ccb4-4ae8-cb6a-2167e2ffdea4"
# Two-sample KS comparison of measured background and experiment gaps
n = len(background2_delta_sorted)
m = len(experiment_delta_sorted)
if n == 0 or m == 0:
    raise ValueError("Both background and experiment need inter-event times.")

# Significance level and large-sample two-sample critical distance
alpha = 0.003
D_alpha = np.sqrt(-0.5 * np.log(alpha / 2)) * np.sqrt((m + n) / (m * n))

# Evaluate both empirical CDFs at every observed gap, including their steps
ks_grid = np.unique(np.concatenate(([0.0], background2_delta_sorted, experiment_delta_sorted)))
background_cdf = np.searchsorted(background2_delta_sorted, ks_grid, side="right") / n
experiment_cdf = np.searchsorted(experiment_delta_sorted, ks_grid, side="right") / m

upper_bound = np.clip(background_cdf + D_alpha, 0, 1)
lower_bound = np.clip(background_cdf - D_alpha, 0, 1)

plt.figure(figsize=(8, 5))
plt.step(ks_grid, background_cdf, where="post", label="Background 2")
plt.step(ks_grid, experiment_cdf, where="post", label=f"Experiment (λ = {experiment_lam:.5f}/s)")
plt.plot(background2_delta_sorted, background2_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {background2_lam:.5f}/s)")
plt.fill_between(ks_grid, lower_bound, upper_bound, step="post", color='gray', alpha=0.3,
                 label="Approx. two-sample KS band (α = 0.003)", zorder=1)

# # Max deviation marker and vertical line
# plt.plot(t_max_dev, cdf_with_synthetic_neutrons[max_index], 'ko', label="KS Statistic (D)")
# plt.vlines(t_max_dev, P_theory_at_synth[max_index], cdf_with_synthetic_neutrons[max_index], color='k', linestyles='dotted')

# # Annotate
# plt.annotate(f"D = {ks_statistic:.4f}", xy=(t_max_dev, cdf_with_synthetic_neutrons[max_index]),
#              xytext=(t_max_dev + 0.2, cdf_with_synthetic_neutrons[max_index] + 0.01),
#              arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)

# Recompute bounds for synthetic burst dataset
#n_synth = len(synthetic_deltas_sorted)
#D_alpha_synth = np.sqrt(-0.5 * np.log(alpha / 2) / n_synth)
#P_poisson_synth = 1 - np.exp(-lam * synthetic_deltas_sorted)
#upper_bound_synth = np.clip(P_poisson_synth + D_alpha_synth, 0, 1)
#lower_bound_synth = np.clip(P_poisson_synth - D_alpha_synth, 0, 1)

#first_cross_index = np.argmax(
#    (cdf_with_synthetic_neutrons > upper_bound_synth) |
#    (cdf_with_synthetic_neutrons < lower_bound_synth)
#)

#plt.axvline(synthetic_deltas_sorted[first_cross_index], color='purple', linestyle=':', label="Statistical significance achieved")


plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t)")
plt.xlim([0,3600])
#plt.ylim([0,0.2])
plt.title(f"Cumulative inter-pulse time probability. PSP > {psp}, Channel > {energy}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<!-- #region id="7DbJ2gPzFBli" -->
Let's compare the soaking period and the laser period.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 89} id="I_MqJuY4FSc6" outputId="1d0c078b-252f-45df-8b3d-68aeae0cb46a"
experiment_times = {'Begin soak': ['2025-04-04 17:37:00'],
        'End soak': ['2025-05-06 19:01:00'],
        'Begin laser': ['2025-05-08 11:00:00'],
        'End laser': ['2025-06-06 12:00:00']}
experiment_time_periods_df = pd.DataFrame(experiment_times)

# Convert the columns to datetime objects
for col in experiment_time_periods_df.columns:
    experiment_time_periods_df[col] = pd.to_datetime(experiment_time_periods_df[col])

display(experiment_time_periods_df)
```

```python id="A2UvXjnMJBoS"
experiment_neutron_data, experiment_neutron_periods = get_all_event_data(experiment_time_periods_df, f">{psp}", f">{energy}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="yMqZq4teJZHy" outputId="f6f9eaee-379f-4fad-e84e-ad550a31515d"
experiment_neutron_data["Begin soak"].head()
```

<!-- #region id="MYUH_NvgJSZS" -->
We have broken apart the neutron data for the soaking period and the laser period. Let's check whether they follow a poisson distribution, and whether the average interneutron time is different for the two periods. We might expect to see some variation because the laser period was conducted in the basement, where the background rate could be different.

We will look at the soaking period first.
<!-- #endregion -->

```python id="u34QbBmmeYLO"
soaking_neutron_data = experiment_neutron_data["Begin soak"]
soaking_deltas = np.diff(soaking_neutron_data.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
soaking_delta_sorted = np.sort(soaking_deltas)
```

```python id="vSM1fmTqgGiu"
# Estimate lambda for soaking period (rate in events per second)
soaking_lam = 1 / np.mean(soaking_delta_sorted)

# Calculate the theoretical distribution
soaking_P_poisson = 1 - np.exp(-soaking_lam * soaking_delta_sorted)

soaking_P_exp = np.arange(1, len(soaking_delta_sorted) + 1) / len(soaking_delta_sorted)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6poBOt52gZ4u" outputId="c41698fe-7d63-4f91-fbfe-10ca64193534"
soaking_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="_fEIfuDDgcRu" outputId="bc7e5388-781e-4072-d729-c1acedcbc142"
plt.figure(figsize=(8, 5))
plt.plot(soaking_delta_sorted, soaking_P_exp, label="Empirical")
plt.plot(soaking_delta_sorted, soaking_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {soaking_lam:.5f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
plt.xlim([0,3600])
plt.title("Cumulative inter-pulse time probability for soaking period")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<!-- #region id="rmuluhbGdvh1" -->
The distribution looks poisson, but the distribution mean looks higher than the calculated mean.

There are a few gaps in the data when data collection was interrupted. We can exclude these gaps and see if these outliers are dragging down the mean.
<!-- #endregion -->

```python id="6q4qfkVdeJ1F"
#Drop soaking deltas greater than 6 hours
soaking_delta_sorted = soaking_delta_sorted[soaking_delta_sorted < 6*60*60]

# Estimate lambda for soaking period (rate in events per second)
soaking_lam = 1 / np.mean(soaking_delta_sorted)

# Calculate the theoretical distribution
soaking_P_poisson = 1 - np.exp(-soaking_lam * soaking_delta_sorted)

soaking_P_exp = np.arange(1, len(soaking_delta_sorted) + 1) / len(soaking_delta_sorted)
```

```python colab={"base_uri": "https://localhost:8080/"} id="7T0dRrofebst" outputId="580ac37f-a818-47f5-bb41-06838a251b7c"
soaking_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="fIGk3cmgednl" outputId="e66492b4-ef5e-43f7-bdea-5685fbdfaaaa"
# Number of samples (n)
n = len(soaking_delta_sorted)

# Desired confidence level (e.g. ~3σ = 99.7%)
alpha = 0.003# Kolmogorov–Smirnov critical value for given alpha and n
D_alpha = np.sqrt(-0.5 * np.log(alpha / 2) / n)


# Upper and lower bounds of the confidence band
upper_bound = np.clip(soaking_P_poisson + D_alpha, 0, 1)
lower_bound = np.clip(soaking_P_poisson - D_alpha, 0, 1)

plt.figure(figsize=(8, 5))
plt.plot(soaking_delta_sorted, soaking_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {soaking_lam:.5f}/s)")
plt.plot(soaking_delta_sorted, soaking_P_exp, label=f"Experiment (λ = {soaking_lam:.5f}/s)")
plt.fill_between(soaking_delta_sorted, lower_bound, upper_bound, color='gray', alpha=0.3, label="~3σ Confidence Band", zorder=1)

#plt.axvline(experiment_delta_sorted[first_cross_index], color='purple', linestyle=':', label="Statistical significance achieved")


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

<!-- #region id="A3xOQ1pJe5cV" -->
With the large gaps excluded, the inter-pulse time for the soaking period is well fit by a Poisson distribution.
<!-- #endregion -->

<!-- #region id="b4FHp65eg4Zf" -->
Now let's check the laser period.
<!-- #endregion -->

```python id="enMQTvRFg8xm"
laser_neutron_data = experiment_neutron_data["Begin laser"]
laser_deltas = np.diff(laser_neutron_data.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
laser_delta_sorted = np.sort(laser_deltas)
```

```python id="-bLSmnkzhH5P"
# Estimate lambda for laser period (rate in events per second)
laser_lam = 1 / np.mean(laser_delta_sorted)

# Calculate the theoretical distribution
laser_P_poisson = 1 - np.exp(-laser_lam * laser_delta_sorted)

laser_P_exp = np.arange(1, len(laser_delta_sorted) + 1) / len(laser_delta_sorted)
```

```python colab={"base_uri": "https://localhost:8080/"} id="idBBHwTohRyH" outputId="ae8c7015-4282-49f8-954c-9ae847de62b5"
laser_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="lmYfXnyhhUJu" outputId="5ceb249f-35da-4b0d-8808-26416b944d08"
# Number of samples (n)
n = len(laser_delta_sorted)

# Desired confidence level (e.g. ~3σ = 99.7%)
alpha = 0.003# Kolmogorov–Smirnov critical value for given alpha and n
D_alpha = np.sqrt(-0.5 * np.log(alpha / 2) / n)


# Upper and lower bounds of the confidence band
upper_bound = np.clip(laser_P_poisson + D_alpha, 0, 1)
lower_bound = np.clip(laser_P_poisson - D_alpha, 0, 1)

plt.figure(figsize=(8, 5))
plt.plot(laser_delta_sorted, laser_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {laser_lam:.5f}/s)")
plt.plot(laser_delta_sorted, laser_P_exp, label=f"Experiment (λ = {laser_lam:.5f}/s)")
plt.fill_between(laser_delta_sorted, lower_bound, upper_bound, color='gray', alpha=0.3, label="~3σ Confidence Band", zorder=1)

#plt.axvline(experiment_delta_sorted[first_cross_index], color='purple', linestyle=':', label="Statistical significance achieved")


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
