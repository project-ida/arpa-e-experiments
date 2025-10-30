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

<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-01/Nassisi_1_Refined_PSD.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-01/Nassisi_1_Refined_PSD.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>

<!-- #region id="sceBesRj1jlf" -->
#Nassisi 1
We have noticed that due to the long duration of the Nassisi experiment, there is some detector drift, causing the neutron and gamma bands to shift over time. In order to get a more accurate count of neutrons, we will break the experiment into chunks and analyze each separately.

Nassisi 1 ran from 16:25 April 4 to 12:00 June 20.
<!-- #endregion -->

```python id="hwEs60h81e1_"
# Auth
import sys, os
import shutil
from google.colab import drive
from google.colab import auth

# Data analysis
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sqlalchemy import create_engine, text

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
```

```python colab={"base_uri": "https://localhost:8080/", "height": 54} id="rVUwPPMo2m91" outputId="b5edc9ea-e599-492c-df14-5f24d99f662c"
# Mount Drive
drive.mount('/content/drive')

# Copy SQL credentials from Google drive
shutil.copy("/content/drive/MyDrive/Nucleonics/.env/psql_credentials_readonly.py", "psql_credentials.py")
```

```python id="G5MWwce52qaD"
# Import SQL credentials
from psql_credentials import PGUSER, PGPASSWORD, PGHOST, PGPORT, PGDATABASE

# Create the database connection string
connection_uri = f'postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}'
engine = create_engine(connection_uri)
```

```python id="8REFFz9g2u9j"
def get_psd_data(start_time, end_time, digitizer, channel_number):
  query = f"""
  SELECT
      width_bucket(channels[1], 0, 1, 128) AS psp_bin,
      width_bucket(channels[2], 0, 4000, 512) AS energy_bin,
      COUNT(*) AS count
  FROM {digitizer}_ch{channel_number}
  WHERE time BETWEEN '{start_time}' AND '{end_time}'
  GROUP BY psp_bin, energy_bin
  ORDER BY psp_bin, energy_bin;
  """
  return pd.read_sql(query, engine, index_col=None)

```

```python id="9RvmXS0OBcaV"
def get_event_data(start_time, end_time, psp=">0", energy=">0"):
  query = f"""
  SELECT * FROM {digitizer}_ch{channel_number}
  WHERE channels[1] {psp} AND channels[2] {energy}
  AND time BETWEEN '{start_time}' AND '{end_time}'
  ORDER BY time;
  """
  df = pd.read_sql(query, engine, index_col=None)
  df.set_index('time', inplace=True)
  return df
```

```python id="jrIKu1-RC3_E"
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

```python id="0_Wu3GWC88Bd"
# --- Gaussian definitions ---
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def double_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
    return (gaussian(x, amp1, mu1, sigma1) +
            gaussian(x, amp2, mu2, sigma2))

def find_psp_threshold_gaussian(data, target_energy=500,
                                     energy_range=(0, 4000), psp_range=(0, 1),
                                     energy_bins=512, psp_bins=128,
                                     plot=True, mean_tol=0.05, amp_ratio_tol=0.1,
                                     num_sigma=4):
    """
    Fits Gaussian(s) to PSP distribution at a given energy and finds PSP threshold
    where the lower psp Gaussian drops to level give by num_sigma*sigma
    Falls back to single Gaussian if fitted Gaussians are too similar.
    """

    # Step 1: Map energy to closest bin
    bin_width_e = (energy_range[1] - energy_range[0]) / energy_bins
    closest_energy_bin = int(round(target_energy / bin_width_e))
    closest_energy_bin = max(1, min(closest_energy_bin, energy_bins))

    # Step 2: Filter data for that energy bin
    filtered = data[data['energy_bin'] == closest_energy_bin]
    if filtered.empty:
        raise ValueError(f"No data for bin {closest_energy_bin} (~{target_energy})")

    # Step 3: Build histogram
    psp_bin_width = (psp_range[1] - psp_range[0]) / psp_bins
    hist = np.zeros(psp_bins)
    for _, row in filtered.iterrows():
        idx = row['psp_bin'] - 1
        if 0 <= idx < psp_bins:
            hist[idx] += row['count']

    psp_centres = np.arange(psp_bins) * psp_bin_width + psp_bin_width / 2

    # Step 4: Try to detect peaks
    peaks, _ = find_peaks(hist, height=np.max(hist) * 0.05)
    use_double = len(peaks) >= 2

    if use_double:
        # Initial guess: amplitudes, means, sigmas
        p0 = [hist[peaks[0]], psp_centres[peaks[0]], 0.01,
              hist[peaks[1]], psp_centres[peaks[1]], 0.01]
        bounds = (0, [np.inf, 1, np.inf, np.inf, 1, np.inf])
        popt, _ = curve_fit(double_gaussian, psp_centres, hist, p0=p0, bounds=bounds)
        params = [(popt[0], popt[1], popt[2]), (popt[3], popt[4], popt[5])]

        # Check if we should fall back to single Gaussian
        mu_diff = abs(params[0][1] - params[1][1])
        amp_ratio = min(params[0][0], params[1][0]) / max(params[0][0], params[1][0])
        if mu_diff < mean_tol or amp_ratio < amp_ratio_tol:
            use_double = False

    if not use_double:
        # Fit single Gaussian
        p0 = [np.max(hist), psp_centres[np.argmax(hist)], 0.01]
        bounds = (0, [np.inf, 1, np.inf])
        popt, _ = curve_fit(gaussian, psp_centres, hist, p0=p0, bounds=bounds)
        params = [tuple(popt)]

    # Step 5: Select lower Gaussian (smallest mean)
    lower_gauss = min(params, key=lambda p: p[1])
    amp, mu, sigma = lower_gauss

    # Step 6: Compute PSP threshold at num_sigma*sigma
    psp_threshold = mu + num_sigma*sigma  # right-side cutoff

    # Step 7: Plot
    if plot:
        plt.figure(figsize=(8, 5))
        plt.bar(psp_centres, hist, width=psp_bin_width, alpha=0.6, label="Data")
        if use_double:
            plt.plot(psp_centres, double_gaussian(psp_centres, *popt), 'k-', label="Double Gaussian fit")
            for amp_i, mu_i, sigma_i in params:
                plt.plot(psp_centres, gaussian(psp_centres, amp_i, mu_i, sigma_i), '--', label=f"Gaussian μ={mu_i:.3f}")
        else:
            plt.plot(psp_centres, gaussian(psp_centres, *params[0]), 'k-', label="Single Gaussian fit")
        plt.axvline(psp_threshold, color='r', linestyle='--', label=f"Threshold={psp_threshold:.3f}")
        plt.xlabel('PSP')
        plt.ylabel('Count')
        plt.title(f'PSP Gaussian Fit at Energy ~{target_energy}')
        plt.legend()
        plt.show()

    return psp_threshold
```

<!-- #region id="VD33ONd-3CG-" -->
Until 11:32 May 8, this experiment was recorded on the 8-channel digitzer: channel 0.
<!-- #endregion -->

```python id="ceL2TYKU213e"
digitizer = "caen8ch"
channel_number = 0
```

```python colab={"base_uri": "https://localhost:8080/"} id="245c4535" outputId="e172f5d0-f4bb-463f-acce-3c92e5b94f93"
from datetime import datetime, timedelta

start_time_str = '2025-04-04 16:25'
end_time_str = '2025-05-08 11:32'

start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')

time_periods = []
all_times = []
current_start_time = start_time

while current_start_time < end_time:
    current_end_time = current_start_time + timedelta(days=2)
    if current_end_time > end_time:
        current_end_time = end_time
    time_periods.append((current_start_time.strftime('%Y-%m-%d %H:%M'), current_end_time.strftime('%Y-%m-%d %H:%M')))
    all_times.append((current_start_time.strftime('%Y-%m-%d %H:%M'), current_end_time.strftime('%Y-%m-%d %H:%M')))
    current_start_time = current_end_time

for start, end in time_periods:
    print(f"Start: {start}, End: {end}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="SOPp_MMO7k54" outputId="14378baa-5ffb-452f-d99d-a690b606d095"
#For each start and end time in start and end, run get_psd_data and return a list of the PSPs
psd_list = []
neutron_rates = []
for start, end in time_periods:
    df_psd = get_psd_data(start, end, digitizer, channel_number)
    psp_threshold = find_psp_threshold_gaussian(df_psd, target_energy=100, num_sigma=4)
    psd_list.append(psp_threshold)
    neutron_data = get_event_data(start,end,f">{psp_threshold}",">500")
    neutron_data = reconstruct_ns_pulses(neutron_data)
    deltas = np.diff(neutron_data.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
    delta_sorted = np.sort(deltas)
    lam = 1 / np.mean(delta_sorted)
    print(lam)
    neutron_rates.append(lam)

    #Construct cumulative probability distributions
    P_poisson = 1 - np.exp(-lam * delta_sorted)
    P_exp = np.arange(1, len(delta_sorted) + 1) / len(delta_sorted)

    #Plot the counts per minute for the time period
    cpm = neutron_data.resample("60s").size().rename("counts").to_frame()
    fig = go.Figure(layout=dict(yaxis_title="Counts per min", showlegend=False, height=600, width=800))
    fig.add_trace(go.Scattergl(name="Counts per min", x=cpm.index, y=cpm.counts))

    #Plot the distribution and compare with Poisson
    plt.figure(figsize=(8, 5))
    plt.plot(delta_sorted, P_exp, label="Empirical")
    plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.2f}/s)")

    plt.xlabel("t (seconds)")
    plt.ylabel("P(Δt ≤ t )")
    plt.title(f"{start} to {end} Cumulative inter-pulse time probability")
    #plt.xlim([0,30])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


```

<!-- #region id="8V3QyKyfjKRz" -->
The first time block deviates from Poisson because it includes the calibration period.
<!-- #endregion -->

<!-- #region id="hFep1Nrn-6s-" -->
We now do the same for the second half, which was recorded on channel 1.
<!-- #endregion -->

```python id="TzXX1scK_CvE"
channel_number = 1
```

```python colab={"base_uri": "https://localhost:8080/"} id="Qv0iD7Y0_I6l" outputId="9216a2a6-ecdc-4021-ebfc-da68e178c5bb"
start_time_str = '2025-05-08 11:32'
end_time_str = '2025-06-20 12:00'

start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')

time_periods = []
current_start_time = start_time

while current_start_time < end_time:
    current_end_time = current_start_time + timedelta(days=2)
    if current_end_time > end_time:
        current_end_time = end_time
    time_periods.append((current_start_time.strftime('%Y-%m-%d %H:%M'), current_end_time.strftime('%Y-%m-%d %H:%M')))
    all_times.append((current_start_time.strftime('%Y-%m-%d %H:%M'), current_end_time.strftime('%Y-%m-%d %H:%M')))
    current_start_time = current_end_time

for start, end in time_periods:
    print(f"Start: {start}, End: {end}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="WWaxJuTV_UXN" outputId="d1150508-8317-4786-b1e9-e25e5eaa107a"
for start, end in time_periods:
    df_psd = get_psd_data(start, end, digitizer, channel_number)
    psp_threshold = find_psp_threshold_gaussian(df_psd, target_energy=100, num_sigma=4)
    psd_list.append(psp_threshold)
    neutron_data = get_event_data(start,end,f">{psp_threshold}",">500")
    neutron_data = reconstruct_ns_pulses(neutron_data)
    deltas = np.diff(neutron_data.index.values).astype("timedelta64[ns]") / np.timedelta64(1, "s")
    delta_sorted = np.sort(deltas)
    lam = 1 / np.mean(delta_sorted)
    print(lam)
    neutron_rates.append(lam)

    #Construct cumulative probability distributions
    P_poisson = 1 - np.exp(-lam * delta_sorted)
    P_exp = np.arange(1, len(delta_sorted) + 1) / len(delta_sorted)

    #Plot the distribution and compare with Poisson
    plt.figure(figsize=(8, 5))
    plt.plot(delta_sorted, P_exp, label="Empirical")
    plt.plot(delta_sorted, P_poisson, linestyle="--", color="red", label=f"Poisson (λ = {lam:.2f}/s)")

    plt.xlabel("t (seconds)")
    plt.ylabel("P(Δt ≤ t )")
    plt.title(f"{start} to {end} Cumulative inter-pulse time probability")
    #plt.xlim([0,30])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

```

```python colab={"base_uri": "https://localhost:8080/"} id="0evnwW6I_fBb" outputId="f0cf4a16-9b2e-46b2-f5f4-266ec941f625"
psd_list
```

```python colab={"base_uri": "https://localhost:8080/"} id="ByO4Kw2rHk4O" outputId="2c0ca9f3-02c5-4e40-96a9-59a152fa740a"
neutron_rates
```

```python colab={"base_uri": "https://localhost:8080/"} id="PLwGhHZBKXF9" outputId="655725b7-ebc7-4189-c98c-dc972c5810f4"
all_times
```

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="i3r1pXsHHps1" outputId="b19b4042-8a3e-4aa3-d22b-ef0f47de4f25"
#Plot neutron_rates over time using pltly
#Convert all_times to timestamps
start_times = [period[0] for period in all_times]

fig = go.Figure(layout=dict(yaxis_title="Neutrons per second", showlegend=False, height=600, width=800))
fig.add_trace(go.Scattergl(name="Neutrons per second", x=start_times, y=neutron_rates))


```

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="rjRTR0bLeDfR" outputId="aa335569-1032-461b-d04d-0e1dc0cc1035"
fig = go.Figure(layout=dict(yaxis_title="PSP", showlegend=False, height=600, width=800))
fig.add_trace(go.Scattergl(name="PSP", x=start_times, y=psd_list))
```

<!-- #region id="GvH1DBGPO2nG" -->
#Conclusion

We find that in fact, the change in the PSP distribution came during a discrete event, namely, when the chamber was transported to the Nelson lab for laser treatment. During transit, the cable transmitting the pulse voltage to the digitizer was damaged and replaced. The different properties of the cable caused a shift in PSD. We therefore split our analysis into before and after the move, and use different PSP cutoffs for each period.
<!-- #endregion -->
