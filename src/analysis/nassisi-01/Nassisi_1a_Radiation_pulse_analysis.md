---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-01/Nassisi_1a_Radiation_pulse_analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-01/Nassisi_1a_Radiation_pulse_analysis.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>

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
from scipy.stats import kstest


#Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
```

<!-- #region id="3AaQ35teouCo" -->
## Authentication

We need to do a few authentication steps:
- Bring in the database credentials from Google drive so that we can pull data from the live database.
- Bring in the nuclear particle master sheet ID
-  Authenticate Colab to pull the nuclear particle master sheet using the Drive API.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XK6IM7v-hnag" outputId="b2defee9-88a3-4855-f604-887edcf000e6"
# Mount Drive
drive.mount('/content/drive')

# Copy SQL credentials from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/psql_credentials_readonly.py", "psql_credentials.py")

# Copy sheet ID file from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/sheet_ids.py", "sheet_ids.py");
```

```python id="OA2IJWNSoyIF"
# Import SQL credentials
from psql_credentials import PGUSER, PGPASSWORD, PGHOST, PGPORT, PGDATABASE

# Import sheet ID for the nuclear particle master sheet
from sheet_ids import NUCLEAR_PARTICLE_MASTER as sheet_id

# Create the database connection string
connection_uri = f'postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}'
engine = create_engine(connection_uri)
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

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="RndMn9Zos-LC" outputId="42d47acb-687f-45fd-90ac-448bc21b7c40"
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
  df = pd.read_sql(query, engine, index_col=None)
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

```python colab={"base_uri": "https://localhost:8080/"} id="XW3JZibgWexj" outputId="ecb75a48-4c2c-423f-e96b-df570b5c93ac"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 175} id="Q3hFn5Z5tXyv" outputId="680cb18b-0bff-4f47-feab-30c5ddfbfb20"
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

```python colab={"base_uri": "https://localhost:8080/"} id="TXshCDTAeGGZ" outputId="5bbd1185-167e-415f-830e-6b3fe0125772"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="qlhkMPk9s-F_" outputId="c83ea066-b3bb-4326-8f39-52220af93932"
plt.figure(figsize=(8, 5))
plt.plot(delta_sorted, P_exp, label="Empirical")
plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.2f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
plt.title("Background 1 cumulative inter-pulse time probability")
plt.xlim([0,30])
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

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="iPSuT_3dJ5kp" outputId="ce3133f2-6c2f-4ff2-e575-75c7afbdb887"
background_cpm = background.resample("60s").size().rename("counts").to_frame()
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=800))
fig.add_trace(go.Scattergl(name="Counts per min", x=background_cpm.index, y=background_cpm.counts))
```

<!-- #region id="B8ZzYbhLlMNV" -->
## Anomaly detection

We can use the inter-pulse cumulative probability to detect deviations from normal background, aka anomalies.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PdHlw300IfEy" outputId="0536b985-754c-4641-d2b9-11f1d53166e0"
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

```python colab={"base_uri": "https://localhost:8080/"} id="R5l7VjXUAJCx" outputId="1482da07-198a-47dd-e7c6-0942732df694"
experiment_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="hXgGNo7aIHsn" outputId="e3128c9f-5e8e-4dd0-a8c2-4330a4b48a12"
plt.figure(figsize=(8, 5))
plt.plot(experiment_delta_sorted, experiment_P_exp, label="Empirical")
plt.plot(experiment_delta_sorted, experiment_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {experiment_lam:.2f}/s)")

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

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="JdyP7cveiSOR" outputId="93bf0ce5-bbe3-4871-c7d2-d42a7106abf7"
experiment_cpm = experiment.resample("60s").size().rename("counts").to_frame()
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=1200))
fig.add_trace(go.Scattergl(name="Counts per min", x=experiment_cpm.index, y=experiment_cpm.counts))
```

<!-- #region id="JZbwIIHfBR_q" -->
We can exclude these spikes:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 634} id="nO9T4l3x1Rmc" outputId="2f1cf7af-d47c-4288-aaa5-ddf86c39c929"
experiment_cpm = experiment.resample("60s").size().rename("counts").to_frame()
#Include only counts greater than 0 and less than 200
experiment_cpm = experiment_cpm[(experiment_cpm.counts < 200) & (experiment_cpm.counts > 0)]
rate_excluding_spikes = np.mean(experiment_cpm.counts)/60
print(f"Rate excluding spike: {rate_excluding_spikes:.2f} events per second")
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=1200))
fig.add_trace(go.Scattergl(name="Counts per min", x=experiment_cpm.index, y=experiment_cpm.counts))
```

<!-- #region id="ITucHJSXjuCr" -->
We can see with the spikes and the gaps exluded, the rate agrees with the background rate we measured prior.
<!-- #endregion -->

<!-- #region id="6McultY7klTK" -->
We can run a 3-sigma significance test and see that the delta deviate from Poisson distributed at low times:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="T57-XEeqCaeR" outputId="86c40a69-fdca-487f-bdfc-ea78e087ef2d"
# Run KS test against an exponential distribution with estimated lambda
ks_stat, p_value = kstest(experiment_delta_sorted, 'expon', args=(0, 1/experiment_lam))

ks_stat, p_value
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="juDzyr-9Cz9r" outputId="765b0ba2-039d-48f2-fbaa-f263c5afab33"
# Number of samples (n)
n = len(experiment_delta_sorted)

# Desired confidence level (e.g. ~3σ = 99.7%)
alpha = 0.003

# Kolmogorov–Smirnov critical value for given alpha and n
D_alpha = np.sqrt(-0.5 * np.log(alpha / 2) / n)


# Upper and lower bounds of the confidence band
upper_bound = np.clip(experiment_P_poisson + D_alpha, 0, 1)
lower_bound = np.clip(experiment_P_poisson - D_alpha, 0, 1)

plt.figure(figsize=(8, 5))
plt.plot(experiment_delta_sorted, experiment_P_exp, label="Empirical")
plt.plot(experiment_delta_sorted, experiment_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {experiment_lam:.2f}/s)")
plt.fill_between(experiment_delta_sorted, lower_bound, upper_bound, color='gray', alpha=0.3, label="~3σ Confidence Band", zorder=1)

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
#plt.xlim([0,5])
#plt.ylim([0,0.2])
plt.title(f"Soaking Period Cumulative inter-pulse time probability. PSP > {psp}, Channel > {energy}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<!-- #region id="Wk6mo9ask3cr" -->
Again, this is due to the spikes: there are roughly 5000 data points included in these spikes with artificially low time separations. Let's drop the last day to exclude the spikes.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="k2m2Fws3OjOz" outputId="f64658ba-3865-4b8f-9969-74039f781898"
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

```python colab={"base_uri": "https://localhost:8080/"} id="hP3QlodTPNfE" outputId="068111bc-274c-4c13-9f80-2e8db56a23c4"
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

```python colab={"base_uri": "https://localhost:8080/"} id="B5RDtAu2PasZ" outputId="f5c89fcf-2997-48e4-81af-b5fec4f21683"
experiment_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="1kWumQ9GPjUU" outputId="327d7941-b004-4b7d-904b-2b378330b406"
# Number of samples (n)
n = len(experiment_delta_sorted)

# Desired confidence level (e.g. ~3σ = 99.7%)
alpha = 0.003

# Kolmogorov–Smirnov critical value for given alpha and n
D_alpha = np.sqrt(-0.5 * np.log(alpha / 2) / n)


# Upper and lower bounds of the confidence band
upper_bound = np.clip(experiment_P_poisson + D_alpha, 0, 1)
lower_bound = np.clip(experiment_P_poisson - D_alpha, 0, 1)

plt.figure(figsize=(8, 5))
plt.plot(experiment_delta_sorted, experiment_P_exp, label="Empirical")
plt.plot(experiment_delta_sorted, experiment_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {experiment_lam:.4f}/s)")
plt.fill_between(experiment_delta_sorted, lower_bound, upper_bound, color='gray', alpha=0.3, label="~3σ Confidence Band", zorder=1)

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
