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

<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-03/Nassisi_3_Radiation_pulse_analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-03/Nassisi_3_Radiation_pulse_analysis.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>

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

```python colab={"base_uri": "https://localhost:8080/"} id="XK6IM7v-hnag" outputId="ed74c418-5a13-43b8-c3ea-987a0acae0fc"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="RndMn9Zos-LC" outputId="9b2e1ad5-d4e7-4f12-b1c0-d66f9b480d3a"
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

```python colab={"base_uri": "https://localhost:8080/"} id="XW3JZibgWexj" outputId="0bc7bea0-3691-4831-c1a9-7380f44ff879"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="Q3hFn5Z5tXyv" outputId="96accbe0-779a-430d-f77b-7c06747b4ec6"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="qlhkMPk9s-F_" outputId="cc3948b4-81bd-4224-c8ad-821f2111917c"
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

```python colab={"base_uri": "https://localhost:8080/"} id="ssCwTHVKiWEg" outputId="1053d0d0-5107-4b1d-eaad-42f512fb5cb2"
background2_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="OEVPdqQEibrg" outputId="7208657c-3f1d-439f-ba60-c84bfb4a5f1b"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="iPSuT_3dJ5kp" outputId="0a8fe806-5410-4219-ea4c-d833c49eb65a"
background_cpm = background.resample("60s").size().rename("counts").to_frame()
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=800))
fig.add_trace(go.Scattergl(name="Counts per min", x=background_cpm.index, y=background_cpm.counts))
```

<!-- #region id="B8ZzYbhLlMNV" -->
## Anomaly detection

We can use the inter-pulse cumulative probability to detect deviations from normal background, aka anomalies.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PdHlw300IfEy" outputId="ff18aaf0-8e0c-49a2-afc8-26c8e0d68760"
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

```python colab={"base_uri": "https://localhost:8080/"} id="R5l7VjXUAJCx" outputId="3a1d4fb9-45ee-4e8e-a456-c04403e2f84b"
experiment_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="hXgGNo7aIHsn" outputId="5acf387f-1f26-458b-81e3-248eb116ec39"
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

```python colab={"base_uri": "https://localhost:8080/"} id="M1BAZ4z5pPeL" outputId="6ba62292-0835-47db-dffe-7ae44039e475"
neutron_periods["Background 1"]
```

```python colab={"base_uri": "https://localhost:8080/"} id="T57-XEeqCaeR" outputId="fd6639f7-905f-47f9-8f35-7fef9351d939"
# Run KS test against an exponential distribution with estimated lambda
ks_stat, p_value = kstest(experiment_delta_sorted, 'expon', args=(0, 1/experiment_lam))

ks_stat, p_value
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="juDzyr-9Cz9r" outputId="31b5d7ec-1aa6-40d4-de07-f2fd98f1e75c"
# Number of samples (n)
n = len(experiment_delta_sorted)

# Desired confidence level (e.g. ~3σ = 99.7%)
alpha = 0.003

# Kolmogorov–Smirnov critical value for given alpha and n
D_alpha = np.sqrt(-0.5 * np.log(alpha / 2) / n)


# Upper and lower bounds of the confidence band
upper_bound = np.clip(background2_P_poisson + D_alpha, 0, 1)
lower_bound = np.clip(background2_P_poisson - D_alpha, 0, 1)

plt.figure(figsize=(8, 5))
plt.plot(background2_delta_sorted, background2_P_exp, label="Background 2")
plt.plot(experiment_delta_sorted, experiment_P_exp, label=f"Experiment (λ = {experiment_lam:.5f}/s)")
plt.plot(background2_delta_sorted, background2_P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {background2_lam:.5f}/s)")
plt.fill_between(background2_delta_sorted, lower_bound, upper_bound, color='gray', alpha=0.3, label="~3σ Confidence Band", zorder=1)

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

```python colab={"base_uri": "https://localhost:8080/", "height": 89} id="I_MqJuY4FSc6" outputId="9fece09d-0dd3-4ffd-e136-2435326634bc"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="yMqZq4teJZHy" outputId="0ea4500a-d3ad-427b-b571-d3b9b892e61b"
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

```python colab={"base_uri": "https://localhost:8080/"} id="6poBOt52gZ4u" outputId="35c98493-68c1-4609-9bd1-d6e0281709ed"
soaking_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="_fEIfuDDgcRu" outputId="900f45f5-c08b-4dca-833d-ed4ecc7a763c"
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

```python colab={"base_uri": "https://localhost:8080/"} id="7T0dRrofebst" outputId="12a4dfc9-9095-4b4c-f5f9-a8440f2c75a5"
soaking_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="fIGk3cmgednl" outputId="0b1a229a-50cd-464b-8411-53faa4b679c9"
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

```python colab={"base_uri": "https://localhost:8080/"} id="idBBHwTohRyH" outputId="ee777fe3-d53e-4721-f604-0b02ceb141d1"
laser_lam
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="lmYfXnyhhUJu" outputId="1fd77d01-d9a5-468e-e674-2db3eff1078c"
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
