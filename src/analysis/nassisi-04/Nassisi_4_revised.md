---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="3f826b5b-1cf7-45b2-9622-89c10dbf1eb2" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-04/Nassisi_4_revised.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-04/Nassisi_4_revised.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
# 2025-04-04 Nassisi 3 Palladium wire
<!-- #endregion -->

<!-- #region id="487e78f6-0666-4d0c-ade0-30403aa31975" -->
A 784 mg Palladium wire is gas loaded with deuterium, in a 0.6 L chamber. There is a 30 day soaking period, where the deuterium is allowed to diffuse through the wire, followed by a 30 day laser irradiation period. During this time, the chamber was relocated from 13-3100 to the basement of building 6, where it was exposed for 1 hour a day to a UV pulsed laser.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6e5640a1-12da-4157-a5e8-5f73f882e6a7" outputId="032744a0-11c6-4476-af68-af38c3434323"
# Makes Libs accessible and runs notebook from same location regardless of whether colab or local
!pip install colocal -q
import colocal
root, branch, cwd = colocal.setup("https://github.com/project-ida/arpa-e-experiments")
```

```python id="a9b070cf-0f22-4946-a040-1860350240d4"
# Libraries and helper functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from IPython.display import Image
from IPython.display import Video
from IPython.display import HTML

# Use our custom helper functions
# - process_data
# - plot_panels
# - plot_panels_with_scatter
# - print_info
# - load_data
from libs.helpers import *

# Necessary for using load_data on password protected data urls
# - authenticate
# - get_credentials
from libs.auth import *
```

```python id="24457467-13a8-466c-a16f-7d7868e7386b"
meta = {
    "descriptor" : "Palladium wire" # This will go into the title of all plots
}
```

<!-- #region id="d1d7c4fc-7df2-4c54-8be1-2750a9071260" -->
## Reading the raw data
<!-- #endregion -->

<!-- #region id="81tl_imIb5oB" -->
### Temperature
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fde663ef-7691-4c50-8a21-df4e77c67d25" outputId="ec0f51fd-6b0c-45ff-f06f-0d80a6e597f6"
# Read the tempearture data
temperature_df = load_data('https://nucleonics.mit.edu/data/csv-files/completed%20arpa-e%20runs/Nassisi4/Nassisi4-2-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 425} id="686ac467-0ef2-48a8-9598-a80f357130a8" outputId="3ad095da-fdfa-49bd-95d9-eff39ddbd207"
# Print out basic description of the data, including any NaNs
print_info(temperature_df)
```

<!-- #region id="EdHD5Uqht1VO" -->
Plot the temperature for the soaking phase
<!-- #endregion -->

```python id="kVZn9OoKt5w-"
soak_start_time = '2025-04-17 13:00:00'
soak_end_time  = '2025-05-19 15:00:00'

soak_temperature_df = temperature_df[soak_start_time:soak_end_time]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 497} id="e3fd770c-4082-432d-85c6-676c0ffdb901" outputId="ffbfd411-5476-424c-fc1e-563f22ba3c51"
plt.figure(figsize=(8, 4))
plt.plot(soak_temperature_df['Ch3 (C)'], label='Ch3 (C)')
plt.plot(soak_temperature_df['Ch5 (C)'], label='Ch5 (C)')
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} Soaking Phase {temperature_df.index[0].date()}")
plt.legend()
plt.show()
```

<!-- #region id="cwgdHyYbwYr3" -->
Plot the temperature for the laser phase
<!-- #endregion -->

```python id="tyK-D1F6wcT_"
laser_start_time = '2025-05-19 19:00:00'
laser_end_time  = '2025-06-23 19:00:00'

laser_temperature_df = temperature_df[laser_start_time:laser_end_time]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="uoWeTbm5xJNh" outputId="7aee2c78-9427-414b-92f1-1cd096eb1099"
plt.figure(figsize=(8, 4))
plt.plot(laser_temperature_df['Ch3 (C)'], label='Ch3 (C)')
plt.plot(laser_temperature_df['Ch5 (C)'], label='Ch5 (C)')
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} Laser Phase {temperature_df.index[0].date()}")
plt.legend()
plt.show()
```

<!-- #region id="tmQwaUluI1tC" -->
#Pressure
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ENtdPJ0QJBTC" outputId="ac42d3ff-35fa-4c5a-fcc9-a489cd9f3a86"
# Read the pressure data
pressure_df = load_data('https://nucleonics.mit.edu/data/csv-files/completed%20arpa-e%20runs/Nassisi4/Nassisi4-1-fullres.csv')
```

```python id="GXRS0x69h5nG"
#Filter pressure_df to include only values of 'Pressure Ch3 (bar)' between 0 and 1.5

pressure_filtered_df = pressure_df[(pressure_df['Pressure Ch3 (bar)'] >= 1.3) & (pressure_df['Pressure Ch3 (bar)'] <= 1.47)]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 457} id="51Hyn8jaJXY6" outputId="b4a2b133-2620-4302-d78d-97de7f786742"
# Print out basic description of the data, including any NaNs
print_info(pressure_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="hVlhjNwNJcpi" outputId="a6fac77c-91d6-4654-8a24-38607c76aa6f"
plt.figure(figsize=(8, 4))
plt.plot(pressure_filtered_df['Pressure Ch3 (bar)'])
plt.xlabel('Time')
plt.ylabel('Relative Pressure (bar)')
plt.ylim(1.2, 1.5)
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {pressure_df.index[0].date()}")
plt.show()
```

<!-- #region id="K762QYOyiB2m" -->
After the soaking period, the chamber was evacuated, and refilled with deuterium gas, following the original protocol used by Nassisi.

This chamber looks more leak tight than the previous runs. On the other hand, the pressure fluctuates a bit more. Let's compare with the ambient pressure in 13-3100 to see if the fluctuations are correlated.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CR7T2-EUiEb-" outputId="e3cc6363-a774-49d6-94c2-f0ed2faafdd9"
ambientpressure_df = load_data('https://nucleonics.mit.edu/data/csv-files/completed%20arpa-e%20runs/Nassisi4/Nassisi4-9-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="MHLlfnNFiJRO" outputId="304038d3-1984-4fa5-d026-b57f5ff09d8c"
# Print out basic description of the data, including any NaNs
print_info(ambientpressure_df)
```

```python id="soozOPjMiLHO"
ambientpressure_df = ambientpressure_df[((ambientpressure_df['Pressure (mbar)']) < 2000) & ((ambientpressure_df['Pressure (mbar)']) > 800)]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="qVfKVlEyibBe" outputId="ae3015f4-3393-4118-f575-b0e3566fc6ea"
plt.figure(figsize=(8, 4))
plt.plot(ambientpressure_df['Pressure (mbar)'])
plt.xlabel('Time')
plt.ylabel('Ambient Pressure (bar)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {ambientpressure_df.index[0].date()}")
plt.show()
```

<!-- #region id="FO_wQ1yzIba4" -->
To estimate the loading from the pressure data, we need to synchronize it with the temperature data.

The `process_data` function below strips the intersecond time resolution from the data in order to place each diagonsitc measurement on a common time index. Any duplicate datapoints that arrise are averaged. Any NaNs that are generated after merging the dataframes are removed by linear interpolation.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 237} id="ENSG03Yuifcm" outputId="16571f97-044d-446d-b157-7c93d83afa57"
combined_df = process_data([temperature_df, pressure_filtered_df, ambientpressure_df], meta)
combined_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 498} id="P947DdxsikkG" outputId="bf7371fd-65e4-4f91-b1be-0dcc425ca8c2"
#Plot Pressure Ch3 (bar) and Pressure (mbar) together
plt.figure(figsize=(8, 4))
plt.plot(combined_df['Pressure Ch3 (bar)'], label='Chamber Pressure (bar)')
plt.plot(combined_df['Pressure (mbar)']/1000, label='Ambient Pressure (bar)')
plt.xlabel('Time')
plt.ylabel('Pressure (bar)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.legend()
plt.show()
```

<!-- #region id="t_ZXrtR2i3OW" -->
With the ambient pressure reading, we can convert the relative pressure gauge reading to an absolute pressure reading.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 237} id="_jnvLPjviiXG" outputId="70afbfaa-b375-4548-f13a-9a9e2acfedc6"
combined_df['Absolute Pressure (bar)'] = combined_df['Pressure Ch3 (bar)'] + combined_df['Pressure (mbar)']/1000
combined_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="P88CcwhaivE2" outputId="4066b42f-7bcc-426d-82d6-623d97ad5bfa"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['Absolute Pressure (bar)'])
plt.xlabel('Time')
plt.ylabel('Absolute Pressure (bar)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {pressure_df.index[0].date()}")
plt.show()
```

<!-- #region id="Kij7QS51i-J-" -->
This also smooths out a lot of the variation in the gauge reading that was due to fluctuations in the ambient pressure (the large discontinuity around 05/16 is when the deuterium was replenished). We'll plot the pressure during the soaking phase.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 462} id="1fecJWCl_r0c" outputId="b89ca131-1699-4477-8dc2-185b37b5d680"
#Pull the pressure data during the soaking period
soak_pressure_df = pressure_df[soak_start_time:soak_end_time]
plt.figure(figsize=(8, 4))
plt.plot(soak_pressure_df['Pressure Ch3 (bar)'])
plt.xlabel('Time')
plt.ylabel('Relative Pressure (bar)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} Soaking Phase {pressure_df.index[0].date()}")
plt.show()

```

<!-- #region id="-egTlch8jHDm" -->
Let's focus on the early loading period.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 512} id="EZxgIKR4i8q2" outputId="03a5a464-4da1-43e5-9d86-4965757a22fe"
load_pressure_df = combined_df['2025-04-17 13:05':'2025-04-19 18:00']
plt.plot(load_pressure_df['Absolute Pressure (bar)'])
plt.xlabel('Time')
plt.ylabel('Absolute Pressure (bar)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {pressure_df.index[0].date()}")
plt.show()
```

<!-- #region id="ea652753-5d8e-4a87-9d2c-868577ff1b70" -->
### Inferring deuterium loading

The deuterium loading is inferred by using the ideal gas law to calculate how many deuterium molecules are present in the gas over time:

$$N_{D_2} = \frac{PV}{k_B T}$$

and associating any changes $\Delta N_{D_2}$ with deuteium entering the lattice. We can then calculate the loading based on the number of lattice atoms $N_{lattice}$, calculated by dividing the wire mass by the atomic mass of Pd:

$$N_{lattice} = 4.437\times 10^{21}$$

The loading is then:

$$2\frac{\Delta N_{D_2}}{N_{lattice}}$$

The factor 2 arrises because a single $D_2$ molecule becomes 2 deuterons once inside the lattice.
<!-- #endregion -->

```python id="3c4e4339-bd7f-4632-9c8d-21aa23729b88"
# Constants
V = 0.6 / 1000  # Volume of the container in m^3
kB = 1.3806503e-23  # Boltzmann constant in J/K
N_lattice = 4.437e21
```

```python id="64f11bfe-a2d9-45b9-8069-3206860299e2"
combined_df['$D_2$ molecules'] = (combined_df['Absolute Pressure (bar)']*1e5 * V) / (kB * (combined_df['Ch3 (C)'] + 273.15))
```

```python id="56c3bcb5-de7a-4a91-8583-3922d6f47f00"
combined_df['D/Pd Loading'] = 2*(combined_df['$D_2$ molecules']['2025-04-17 13:05:00'] - combined_df['$D_2$ molecules']) / N_lattice
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="6d44a2b5-ddd6-4024-b281-dbd293f6c85c" outputId="bb605730-2810-497e-9f84-fcdba6b281cc"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['$D_2$ molecules'])
plt.xlabel('Time')
plt.ylabel('$D_2$ molecules')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="6abfec40-461f-46e7-a987-00218be5b56f" outputId="4c5c8182-811b-49d5-fd3b-4f668400041b"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['D/Pd Loading'])
plt.xlabel('Time')
plt.ylabel('D/Pd Loading')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

<!-- #region id="Y_xkn3DUmWWB" -->
Again, focusing only on the early loading period:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="PaXfZZnYmaAR" outputId="e06eedeb-a3eb-43db-d205-2bce710c13f7"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['D/Pd Loading']['2025-04-17 13:05':'2025-04-19 18:00'])
plt.xlabel('Time')
plt.ylabel('D/Pd Loading')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()

```

<!-- #region id="p1i_EtyG_m9G" -->
We can fit the loading to a exponential dependence. The justification follows from application of Fick's second law: for an initially hydrogen free cylinder of radius R, and for a fixed boundary concentration C(0,t) = C$_s$, the concentration at depth r from the surface is
$$C(r,t)=C_s[1-2∑_{n=1}^∞\frac{J_0(\alpha_nr/R)}{\alpha_nJ_1(\alpha_n)})\mathrm{exp}(-\frac{\alpha_n^2Dt}{R^2})]$$
where D is the diffusion coefficient, $J_0$ and $J_1$ are Bessel functions of the first and second kind, and $\alpha_n$ are the positive zeros of $J_0$. By integrating the concentration over the whole cylinder, we arrive at the average concentration of hydrogen
$$C(t)=C_s[1-4\sum_{0}^{∞}\frac{1}{\alpha_n^2}\mathrm{exp}(-\frac{\alpha_n^2Dt}{R^2})]$$

At long times, the first term dominates:
$$C(t)\approx C_s[1-\frac{4}{\alpha_1^2}\mathrm{exp}(-\frac{\alpha_1^2Dt}{R^2})]\approx C_s[1-0.692e^{-23.132Dt}]$$

for R = 0.5 mm. For a typical diffusivity of 10$^{-5}$ mm$^2$/s, this is the regime where $t \gg \tau_1 = \frac{R^2}{\alpha_1^2D} \approx 4300$ seconds.

To fit the bulk loading curve, we subtract off the concentration at the surface.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 624} id="GQgKqpIM0bit" outputId="b8980955-41e8-4793-c028-c910422af7fb"

def exp_dependence(t, A, D):
    # This modified function starts at 0 for t=0 and approaches 0.692A as t -> infinity. A is the number of deuterium atoms on the surface, which we assume to be constant.
    return 0.692 * A * (1 -  np.exp(-23.132 * D * t))

# Get the y-data (pressure) and x-data (time index)
y_data = 2*(combined_df['$D_2$ molecules']['2025-04-17 13:05:00']-combined_df['$D_2$ molecules']['2025-04-17 13:05':'2025-04-19 18:00']) #factor of 2 to convert D2 molecules to D atoms

# Convert y_data index to seconds elapsed after the initial value
x_data = (y_data.index - y_data.index[0]).total_seconds().to_numpy()

# Initial guess for A (max y_data) and D (small positive value)
p0 = [1E21, .00001] # A, D in mm^2/s

# Perform the curve fit
try:
  #Here we fit to the later data where the approximation is supposed to hold.
    params, covariance = curve_fit(exp_dependence, x_data[x_data>10000], y_data[(y_data.index - y_data.index[0]).total_seconds().to_numpy()>10000], p0=p0)
    A_fit, D_fit = params
    print(f"Fitted parameters: A={A_fit}, D={D_fit}")

    # Generate fitted data
    y_fit = exp_dependence(x_data, A_fit, D_fit)

    # Plot the original data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, label='Background-subtracted Pressure Data')
    plt.plot(x_data, y_fit, label=f'Fitted Curve: A$(1-0.692e^{{-23.132Dt}})$', linestyle='--') # Updated LaTeX escape sequence and label
    plt.xlabel('Time (seconds)')
    plt.ylabel('D atoms loaded') # Corrected y-axis label
    plt.xticks(rotation=45)
    plt.title(f"{meta['descriptor']} {load_pressure_df.index[0].date()} Background Subtracted Loading Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except RuntimeError as e:
    print(f"Error fitting curve: {e}. Try adjusting initial parameters (p0).")
```

<!-- #region id="mT7cL2QyYMD_" -->
There may be several factors confounding the model. Phase change kinetics may hinder the motion of the deuterium, causing a deviation from simple diffusion behavior. There may also be surface contaminants impeding the uptake of deuterium; the timescale for loading is much longer than typical, taking several days vs. the usual hours. Higher order terms may have contributions that we are neglecting in this approximation.
<!-- #endregion -->

```python id="PgzDU4j35YpK"

```
