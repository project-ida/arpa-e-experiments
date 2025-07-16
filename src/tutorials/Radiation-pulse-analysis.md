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

<!-- #region id="2a9ece45-8fed-412c-b192-4fb1145b4d1b" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/radiation-bin-indepedent/tutorials/Radiation-pulse-analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/radiation-bin-indepedent/tutorials/Radiation-pulse-analysis.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
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
## Running this notebook

Go ahead and change the `experiment_id` below and then run the whole notebook.

You will be asked a couple of times to authenticate with your Google account, but after that all the analysis will happen automatically.
<!-- #endregion -->

```python id="n_BC5wa_rq9S"
experiment_id = 1
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
- Bring in the nuclear particle master sheet ID.
-  Authenticate Colab to pull the nuclear particle master sheet using the Drive API.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XK6IM7v-hnag" outputId="43c4a0f4-83e5-473b-8fa7-465e1844d5c4"
# Mount Drive
drive.mount('/content/drive')

# Copy SQL credentials from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/psql_credentials.py", "psql_credentials.py")

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

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="RndMn9Zos-LC" outputId="c8396f3d-ca5a-4ec1-9e24-e23ca61a2258"
# Find the row where Experiment ID matches
row = df[df['Experiment ID'] == experiment_id]

# Exract digitizer, either 4 channel or 8
digitizer = row["Digitizer"].iloc[0]

# Extract the channel number
channel_number = row["Channel number"].iloc[0]

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

The [simplest PSD analysis](https://github.com/project-ida/arpa-e-experiments/blob/main/tutorials/PSD_Analysis.ipynb) is to use a constant PSP discriminator value above which the pulses are considered to be neutrons, below are gammas. We can also combine this with an similar energy discriminator.

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
<!-- #endregion -->

<!-- #region id="EBdSq9e36GZA" -->
We can extract only the neutron events by using the psp values stored in the master spreadsheet.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XW3JZibgWexj" outputId="da368a50-0965-4149-9686-bbb63082a5a2"
psp
```

```python id="cqnvIHxhtdTt"
neutron_data, neutron_periods = get_all_event_data(times, f">{psp}")
```

<!-- #region id="qYRio4htXHRL" -->
### Reconstructing the pulses
<!-- #endregion -->

<!-- #region id="sUGMTuAstdD_" -->
Let's see what the pulse data looks like for the background.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="Q3hFn5Z5tXyv" outputId="565ae2a9-8f60-49fb-a244-8e031bc7c22a"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="qlhkMPk9s-F_" outputId="68c8122c-4c6b-4762-807d-744b25be281e"
plt.figure(figsize=(8, 5))
plt.plot(delta_sorted, P_exp, label="Empirical")
plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.2f}/s)")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t )")
plt.title("Cumulative inter-pulse time probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

<!-- #region id="jt_FGND9ky3u" -->
Visually, the level of agreement is superb. Let's be more quantitative.

**TODO: Statistical test**
<!-- #endregion -->

<!-- #region id="eJ5bN8PjCQga" -->
It's instructive to look at the cumulative pulses alongside the counts per minute.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="iPSuT_3dJ5kp" outputId="ac93e5a9-c985-494c-9c3a-b7e12df57a3e"
background_cpm = background.resample("60s").size().rename("counts").to_frame()
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=800))
fig.add_trace(go.Scattergl(name="Counts per min", x=background_cpm.index, y=background_cpm.counts))
```

<!-- #region id="B8ZzYbhLlMNV" -->
## Anomaly detection

We can use the inter-pulse cumulative probability to detect deviations from normal background, aka anomalies. To demonstrate this, we'll combine the background period with a calibration period in which a strong neutron source is present.
<!-- #endregion -->

<!-- #region id="lZyl5Rf1D5GD" -->
Support for third party widgets will remain active for the duration of the session. To disable support:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3caxZrbEpInb" outputId="ed9a0ccb-e5eb-41c7-c57b-d6d6701d5c24"
neutron_periods["Calibration"]
```

```python colab={"base_uri": "https://localhost:8080/"} id="M1BAZ4z5pPeL" outputId="24657e4e-f86e-4eba-c73e-f40da1039674"
neutron_periods["Background 1"]
```

<!-- #region id="0VTPZRIopS01" -->
Let's see what a 16 hour "blast" of neutrons does to 36 days worth of background. We'll plot the new cumulative distribution of events alongside the others.
<!-- #endregion -->

```python id="Y_3ULW4uVbRU"
neutrons_with_source =  neutron_data["Calibration"] + neutron_data["Background 1"]
```

```python id="DLprp2FzHSEE"
deltas_with_source = np.diff(neutrons_with_source.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
delta_sorted_with_source = np.sort(deltas_with_source)
cdf_with_source = np.arange(1, len(delta_sorted_with_source) + 1) / len(delta_sorted_with_source)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="pK3mugXcVZ14" outputId="7ec6526e-8e6d-4acf-da43-f332f6b89d05"
plt.figure(figsize=(8, 5))
plt.plot(delta_sorted, P_exp, label="Empirical")
plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.2f}/s)")
plt.plot(delta_sorted_with_source, cdf_with_source, label="Empirical with source")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t)")
plt.title("Cumulative inter-pulse time probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<!-- #region id="QAKyf3zDps1T" -->
We can clearly see the difference between what's expected from a normal background and a strong neutron source.

The reason for the inital jump in the empirical plot with a source is that the source lowers the time between events because there are more events per second. The longer the source is present, the higher the fraction of events that occur within a small time of one another.

Let's again look at the counts per minute associated with this plot.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="9nebqMxqMJwN" outputId="971b7a21-65ec-4d52-9e72-ed027c1db49c"
neutrons_with_source_cpm = neutrons_with_source.resample("60s").size().rename("counts").to_frame()
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=800))
fig.add_trace(go.Scattergl(name="Counts per min", x=neutrons_with_source_cpm.index, y=neutrons_with_source_cpm.counts))
```

<!-- #region id="3yFox81KrMOq" -->
### Synthetic anomalies

Although statistical analysis is the most robust way of determining anomalies, it's important to consider how easy it is to visually discern changes in the distribution because we often lean on visuals in an initial analysis.

For this exercise, we'll simulate the appearance of random short bursts of neutrons and look at the changes to the inter-pulse cumulative distribution.

We'll create a function that allows us to inject a number of poisson distributed bursts of a given duration and strength into an experimetnal period. We'll define the strength through a multiplier which scales the average rate from the given experimental period.
<!-- #endregion -->

```python id="z9X--46fp5Jh"
def inject_poisson_bursts(df, n_bursts=3, burst_duration_s=1.0, burst_multiplier=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Compute background rate
    deltas = np.diff(df.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
    lambda_bg = 1 / np.mean(deltas)

    # Step 2: Pick burst start times uniformly in range
    start_time = df.index.min()
    end_time = df.index.max() - pd.Timedelta(seconds=burst_duration_s)
    burst_starts = pd.to_datetime(
        np.random.uniform(start_time.value, end_time.value, n_bursts).astype("int64")
    )

    all_burst_timestamps = []

    for t0 in burst_starts:
        t0 = pd.Timestamp(t0)
        t1 = t0 + pd.Timedelta(seconds=burst_duration_s)
        lambda_burst = lambda_bg * burst_multiplier
        expected_count = np.random.poisson(lam=lambda_burst * burst_duration_s)

        # Generate inter-pulse times
        inter_arrivals = np.random.exponential(1 / lambda_burst, expected_count)
        event_times = t0 + pd.to_timedelta(np.cumsum(inter_arrivals), unit="s")
        event_times = event_times[event_times <= t1]

        all_burst_timestamps.extend(event_times)

    # Step 3: Create burst DataFrame
    burst_df = pd.DataFrame(index=pd.to_datetime(all_burst_timestamps))
    burst_df["counts"] = 1

    # Step 4: Combine with original
    combined = pd.concat([df, burst_df])
    combined.sort_index(inplace=True)

    return combined
```

<!-- #region id="5nX-L8qCuQ5u" -->
Let's create 10 bursts, with a strength 100 times that of the background and let's make them last for 10 mins each.
<!-- #endregion -->

```python id="IczdqmZ3p5Hn"
neutrons_synthetic = inject_poisson_bursts(background,n_bursts=5, burst_duration_s=600, burst_multiplier=20)
```

```python id="Y8K5OP2cp27n"
synthetic_deltas = np.diff(neutrons_synthetic.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
synthetic_deltas_sorted = np.sort(synthetic_deltas)
cdf_with_synthetic_neutrons = np.arange(1, len(synthetic_deltas_sorted) + 1) / len(synthetic_deltas_sorted)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="1QqaBSlwt0bZ" outputId="b4545369-ad27-4b01-b0c0-dbe2e357c3e4"
plt.figure(figsize=(8, 5))
plt.plot(delta_sorted, P_exp, label="Empirical")
plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.2f}/s)")
plt.plot(synthetic_deltas_sorted, cdf_with_synthetic_neutrons, label="Empirical with synthetic bursts")

plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t)")
plt.xlim([0,5])
plt.ylim([0,0.2])
plt.title("Cumulative inter-pulse time probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<!-- #region id="wM2zwA3CMxhr" -->
Let's again look at the counts per minute that go along with this
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="UobrS_ukM1xM" outputId="242e6c08-a667-4fc6-88de-4be7e24effde"
neutrons_synthetic_cpm = neutrons_synthetic.resample("60s").size().rename("counts").to_frame()
fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=800))
fig.add_trace(go.Scattergl(name="Counts per min", x=neutrons_synthetic_cpm.index, y=neutrons_synthetic_cpm.counts))
```

<!-- #region id="K9D02SUvv78g" -->
The next step is to use some statistical analysis of these synthetic bursts in order to understand what kind of anomalies we might be able to detect within a given degree of certainty.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dMc45lIj-lo-" outputId="e3520a54-fd82-454e-f4e4-f1e6ef96bbe1"
# Run KS test against an exponential distribution with estimated lambda
ks_stat, p_value = kstest(synthetic_deltas_sorted, 'expon', args=(0, 1/lam))

ks_stat, p_value
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="fehl9N_aMwOE" outputId="3522b934-98e8-4c07-aa01-dcb3e2cb94dd"
# Number of samples (n)
n = len(delta_sorted)

# Desired confidence level (e.g. ~3σ = 99.7%)
alpha = 0.003

# Kolmogorov–Smirnov critical value for given alpha and n
D_alpha = np.sqrt(-0.5 * np.log(alpha / 2) / n)


# Upper and lower bounds of the confidence band
upper_bound = np.clip(P_poisson + D_alpha, 0, 1)
lower_bound = np.clip(P_poisson - D_alpha, 0, 1)

plt.figure(figsize=(8, 5))
plt.plot(delta_sorted, P_exp, label="Empirical")
plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.2f}/s)")
plt.plot(synthetic_deltas_sorted, cdf_with_synthetic_neutrons, label="Empirical with synthetic bursts")
plt.fill_between(delta_sorted, lower_bound, upper_bound, color='gray', alpha=0.3, label="~3σ Confidence Band", zorder=1)

# # Max deviation marker and vertical line
# plt.plot(t_max_dev, cdf_with_synthetic_neutrons[max_index], 'ko', label="KS Statistic (D)")
# plt.vlines(t_max_dev, P_theory_at_synth[max_index], cdf_with_synthetic_neutrons[max_index], color='k', linestyles='dotted')

# # Annotate
# plt.annotate(f"D = {ks_statistic:.4f}", xy=(t_max_dev, cdf_with_synthetic_neutrons[max_index]),
#              xytext=(t_max_dev + 0.2, cdf_with_synthetic_neutrons[max_index] + 0.01),
#              arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)

# Recompute bounds for synthetic burst dataset
n_synth = len(synthetic_deltas_sorted)
D_alpha_synth = np.sqrt(-0.5 * np.log(alpha / 2) / n_synth)
P_poisson_synth = 1 - np.exp(-lam * synthetic_deltas_sorted)
upper_bound_synth = np.clip(P_poisson_synth + D_alpha_synth, 0, 1)
lower_bound_synth = np.clip(P_poisson_synth - D_alpha_synth, 0, 1)

first_cross_index = np.argmax(
    (cdf_with_synthetic_neutrons > upper_bound_synth) |
    (cdf_with_synthetic_neutrons < lower_bound_synth)
)

plt.axvline(synthetic_deltas_sorted[first_cross_index], color='purple', linestyle=':', label="Statistical significance achieved")


plt.xlabel("t (seconds)")
plt.ylabel("P(Δt ≤ t)")
plt.xlim([0,4])
plt.ylim([0,0.2])
plt.title("Cumulative inter-pulse time probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

```python id="Z2Ova3eyUA8P"

```
