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
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-01/Nassisi_1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/nassisi-01/Nassisi_1.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
# 2025-04-04 Nassisi 1 Palladium wire
<!-- #endregion -->

<!-- #region id="487e78f6-0666-4d0c-ade0-30403aa31975" -->
A 900 mg Palladium wire is gas loaded with deuterium, in a 0.6 L chamber. There is a 30 day soaking period, where the deuterium is allowed to diffuse through the wire, followed by a 30 day laser irradiation period. During this time, the chamber was relocated from 13-3100 to the basement of building 6, where it was exposed for 1 hour a day to a UV pulsed laser.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6e5640a1-12da-4157-a5e8-5f73f882e6a7" outputId="f3b28500-292f-4ea1-f6e4-ef02d00e3d11"
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

```python colab={"base_uri": "https://localhost:8080/"} id="fde663ef-7691-4c50-8a21-df4e77c67d25" outputId="64170ae6-af69-4946-c9eb-ab9853869bf1"
# Read the tempearture data
temperature_df = load_data('https://nucleonics.mit.edu/data/csv-files/completed%20arpa-e%20runs/Nassisi1/Nassisi1-2-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 457} id="686ac467-0ef2-48a8-9598-a80f357130a8" outputId="42c64901-de29-4e28-d6bb-acc88755768b"
# Print out basic description of the data, including any NaNs
print_info(temperature_df)
```

<!-- #region id="EdHD5Uqht1VO" -->
Plot the temperature for the soaking phase
<!-- #endregion -->

```python id="kVZn9OoKt5w-"
soak_start_time = '2025-04-04 12:00:00'
soak_end_time  = '2025-05-07 12:00:00'

soak_temperature_df = temperature_df[soak_start_time:soak_end_time]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 497} id="e3fd770c-4082-432d-85c6-676c0ffdb901" outputId="573308a3-5b78-425f-9f77-d90b0dc7fae4"
plt.figure(figsize=(8, 4))
plt.plot(soak_temperature_df['Ch3 (C)'], label='Ch3 (C)')
plt.plot(soak_temperature_df['Ch5 (C)'], label='Ch5 (C)')
plt.plot(soak_temperature_df['Ch1 (C) ambient'], label='Ch1 (C) ambient')
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
laser_start_time = '2025-05-07 12:00:00'
laser_end_time  = '2025-06-20 12:00:00'

laser_temperature_df = temperature_df[laser_start_time:laser_end_time]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="uoWeTbm5xJNh" outputId="af28e265-47bd-44b7-a27e-05b9db5b7568"
plt.figure(figsize=(8, 4))
plt.plot(laser_temperature_df['Ch3 (C)'], label='Ch3 (C)')
plt.plot(laser_temperature_df['Ch5 (C)'], label='Ch5 (C)')
plt.plot(laser_temperature_df['Ch1 (C) ambient']-3, label='Ch1 (C) ambient (offset)')
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} Laser Phase {temperature_df.index[0].date()}")
plt.legend()
plt.show()
```

<!-- #region id="hCUyqwka8Sdv" -->
Above, the ambient temperature was shifted down 3 degrees to more clearly show the chamber internal temperature readings. The regular series of spikes show the heating due to the laser treatment.
<!-- #endregion -->

<!-- #region id="tmQwaUluI1tC" -->
#Pressure
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ENtdPJ0QJBTC" outputId="9fbe8fbd-af16-4bab-dd0a-b54e1c5bb305"
# Read the pressure data
pressure_df = load_data('https://nucleonics.mit.edu/data/csv-files/completed%20arpa-e%20runs/Nassisi1/Nassisi1-1-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 520} id="51Hyn8jaJXY6" outputId="07da5cf1-e14e-42e7-d256-546ab99a2f89"
# Print out basic description of the data, including any NaNs
print_info(pressure_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="hVlhjNwNJcpi" outputId="3bef28e7-cf34-49bb-dbdf-3be4b34b1cc9"
plt.figure(figsize=(8, 4))
plt.plot(pressure_df['Pressure Ch3 (bar)'])
plt.xlabel('Time')
plt.ylabel('Relative Pressure (bar)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {pressure_df.index[0].date()}")
plt.show()
```

<!-- #region id="mnm56eYD9VT3" -->
After the soaking period, the chamber was evacuated, and refilled with deuterium gas, following the original protocol used by Nassisi.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 462} id="1fecJWCl_r0c" outputId="af5c810c-bb7a-4409-fe40-6c732ba7008a"
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

<!-- #region id="gTx6hlVVCZm1" -->
It appears that there was a leak. Let's try to correct this background before we look more closely at the loading curve.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 497} id="w_7LNIdgCjyt" outputId="305a5807-c397-49f3-c266-2fceb99c09f2"
#Fit the soaking pressure data from April 7 to May 5 to a line
y = soak_pressure_df['Pressure Ch3 (bar)']['2025-04-07 00:00:00':'2025-05-05 00:00:00']
x = np.arange(len(y))
m, b = np.polyfit(x, y, 1)
plt.figure(figsize=(8, 4))
plt.plot(soak_pressure_df['Pressure Ch3 (bar)'], label='Raw Data')
plt.plot(y.index, m*x+b, label='Fitted Line')
plt.xlabel('Time')
plt.ylabel('Relative Pressure (bar)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} Soaking Phase {pressure_df.index[0].date()}")
plt.legend()
plt.show()

```

```python colab={"base_uri": "https://localhost:8080/", "height": 554} id="GrJJftnpF5Fz" outputId="e467a071-37eb-4804-963e-b351650f4dd0"
load_pressure_df = pressure_df['2025-04-04 17:50:00':'2025-04-05 12:00:00']
#Subtract background m*x+b from load_pressure_df['Pressure Ch3 (bar)']
x = np.arange(len(load_pressure_df['Pressure Ch3 (bar)']))
load_pressure_df['Pressure Ch3 (bar)'] = load_pressure_df['Pressure Ch3 (bar)'] - (m*x) + 1 #adding 1 to convert from relative pressure to absolute
plt.figure(figsize=(8, 4))
plt.plot(load_pressure_df['Pressure Ch3 (bar)'])
plt.xlabel('Time')
plt.ylabel('Absolute Pressure (bar)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {pressure_df.index[0].date()} Background subtracted")
plt.show()

```

<!-- #region id="FO_wQ1yzIba4" -->
To estimate the loading from the pressure data, we need to synchronize it with the temperature data.

The `process_data` function below strips the intersecond time resolution from the data in order to place each diagonsitc measurement on a common time index. Any duplicate datapoints that arrise are averaged. Any NaNs that are generated after merging the dataframes are removed by linear interpolation.
<!-- #endregion -->

```python id="KeGNg5xkIkHV"
combined_df = process_data([temperature_df, load_pressure_df], meta)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="B0NzgLPaI2MI" outputId="edac407f-33cf-43be-f482-7b9fbc685896"
combined_df.head()
```

<!-- #region id="ea652753-5d8e-4a87-9d2c-868577ff1b70" -->
### Inferring deuterium loading

The deuterium loading is inferred by using the ideal gas law to calculate how many deuterium molecules are present in the gas over time:

$$N_{D_2} = \frac{PV}{k_B T}$$

and associating any changes $\Delta N_{D_2}$ with deuteium entering the lattice. We can then calculate the loading based on the number of lattice atoms $N_{lattice}$, calculated by dividing the wire mass by the atomic mass of Pd:

$$N_{lattice} = 5.093\times 10^{21}$$

The loading is then:

$$2\frac{\Delta N_{D_2}}{N_{lattice}}$$

The factor 2 arrises because a single $D_2$ molecule becomes 2 deuterons once inside the lattice.
<!-- #endregion -->

```python id="3c4e4339-bd7f-4632-9c8d-21aa23729b88"
# Constants
V = 0.6 / 1000  # Volume of the container in m^3
kB = 1.3806503e-23  # Boltzmann constant in J/K
N_lattice = 5.093e21
```

```python id="64f11bfe-a2d9-45b9-8069-3206860299e2"
combined_df['$D_2$ molecules'] = (combined_df['Pressure Ch3 (bar)']*1e5 * V) / (kB * (combined_df['Ch3 (C)'] + 273.15))
```

```python id="56c3bcb5-de7a-4a91-8583-3922d6f47f00"
combined_df['D/Pd Loading'] = 2*(combined_df.iloc[0]['$D_2$ molecules'] - combined_df['$D_2$ molecules']) / N_lattice
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="6d44a2b5-ddd6-4024-b281-dbd293f6c85c" outputId="e296e40a-edca-454a-9e80-e773806ea6e7"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['$D_2$ molecules'])
plt.xlabel('Time')
plt.ylabel('$D_2$ molecules')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="6abfec40-461f-46e7-a987-00218be5b56f" outputId="f1e41005-8eaa-416b-88aa-66fab97bc047"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['D/Pd Loading'])
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

```python colab={"base_uri": "https://localhost:8080/", "height": 679} id="GQgKqpIM0bit" outputId="caebdb3b-fb7c-4c3d-8a84-506b9b4e1a7c"

def exp_dependence(t, A, D):
    # This modified function starts at 0 for t=0 and approaches 0.692A as t -> infinity. A is the number of deuterium atoms on the surface, which we assume to be constant.
    return 0.692 * A * (1 - np.exp(-23.132 * D * t))

# Get the y-data (pressure) and x-data (time index)
y_data = 2*(combined_df['$D_2$ molecules'][0]-combined_df['$D_2$ molecules']) #factor of 2 to convert D2 molecules to D atoms

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
    plt.plot(x_data, y_fit, label=f'Fitted Curve: 0.692A$(1-e^{{-23.132Dt}})$', linestyle='--') # Updated LaTeX escape sequence and label
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
There may be several factors confounding the model. The chamber leak might not have been completed accounted for in the background subtraction. We do not capture the saturation value in this dataset. There may also be surface contaminants impeding the uptake of deuterium; the timescale for loading is much longer than typical, taking several days vs. the usual hours. Higher order terms may have contributions that we are neglecting in this approximation.  

<!-- #endregion -->
