---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="3f826b5b-1cf7-45b2-9622-89c10dbf1eb2" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/2024-08-29-palladium-foil.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/2024-08-29-palladium-foil.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
# 2024-08-29 Palladium foil
<!-- #endregion -->

<!-- #region id="487e78f6-0666-4d0c-ade0-30403aa31975" -->
A xy mg Palladium foil is gas loaded with deuterium, in a 0.19L chamber.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6e5640a1-12da-4157-a5e8-5f73f882e6a7" outputId="7c01eff9-1def-4d03-aea8-523bf386a20b"
# RUN THIS IF YOU ARE USING GOOGLE COLAB
import sys
import os
!git clone https://github.com/project-ida/arpa-e-experiments.git
sys.path.insert(0,'/content/arpa-e-experiments')
os.chdir('/content/arpa-e-experiments')
```

```python id="a9b070cf-0f22-4946-a040-1860350240d4"
# Libraries and helper functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image
from IPython.display import Video
from IPython.display import HTML

# Use our custom helper functions
# - process_data
# - plot_panels
# - plot_panels_with_scatter
# - print_info
from libs.helpers import *
```

```python id="24457467-13a8-466c-a16f-7d7868e7386b"
meta = {
    "descriptor" : "Palladium foil" # This will go into the title of all plots
}
```

<!-- #region id="d1d7c4fc-7df2-4c54-8be1-2750a9071260" -->
## Reading the raw data
<!-- #endregion -->

<!-- #region id="81tl_imIb5oB" -->
### Temperature
<!-- #endregion -->

```python id="fde663ef-7691-4c50-8a21-df4e77c67d25"
# Read the tempearture data
temperature_df = pd.read_csv(
    'http://nucleonics.mit.edu/csv-files/loading%20deloading%20runs/thermocouples_september-1.csv',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="686ac467-0ef2-48a8-9598-a80f357130a8" outputId="908f4588-e22f-4b8c-cf8c-4ab4f58149f0"
# Print out basic description of the data, including any NaNs
print_info(temperature_df)
```

<!-- #region id="I5nT2hx4lFdd" -->
`Thermocouple1Ch4` was offline for the whole experiment so we'll drop it.

`Thermocouple1Ch3` is missing a large chunk of data and it's not necessary for the current analysis, so we'll also drop it.

`Thermocouple1Ch2` has one NaN which we'll fix this during the processing stage.
<!-- #endregion -->

```python id="soi2JWUplH5O"
temperature_df.drop('Thermocouple1Ch4', axis=1, inplace=True) # Drop Thermocouple1Ch4
temperature_df.drop('Thermocouple1Ch3', axis=1, inplace=True) # Drop Thermocouple1Ch3
```

<!-- #region id="6e7a2688-aa5a-4533-a75e-f86815179b75" -->
Since we'll only be interested in `Thermocouple1Ch1`, we'll rename it to make plotting a bit easier later.
<!-- #endregion -->

```python id="4dbcfc48-cbc4-40c6-8916-cc8dee41463a"
temperature_df.rename(columns={'Thermocouple1Ch1': 'Temperature (C)'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="e3fd770c-4082-432d-85c6-676c0ffdb901" outputId="bf4d27a1-dd24-42a5-a0f7-8cb4e9ce85db"
plt.figure(figsize=(8, 4))
plt.plot(temperature_df['Temperature (C)'])
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {temperature_df.index[0].date()}")
plt.show()
```

<!-- #region id="75744b06-6ec3-49cc-a1db-8d7b075b176f" -->
### Heating power
<!-- #endregion -->

```python id="3a6692b1-462c-43de-83de-d31864a847b3"
# Read the heating power data
heating_df = pd.read_csv(
    'http://nucleonics.mit.edu/csv-files/loading%20deloading%20runs/thermocouples_september-2.csv',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="81174345-4f9b-4095-ab2e-3dfd172ea8a2" outputId="b24189d5-f0ed-4350-c709-1178d5b90893"
# Print out basic description of the data, including any NaNs
print_info(heating_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="f72ac15c-1a1f-4855-b3ea-eff6979553ce" outputId="dcebcccf-6a4b-48c8-bac8-e07c55572712"
plt.figure(figsize=(8, 4))
plt.plot(heating_df['Voltage'], label="Voltage (V)")
plt.plot(heating_df['Current'], label="Current (A)")
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.legend()
plt.title(f"{meta['descriptor']} {heating_df.index[0].date()}")
plt.show()
```

<!-- #region id="KwGr1eAUcY_h" -->
### Pressure
<!-- #endregion -->

```python id="D8kU4YY7b3W7"
# Read the pressure data
pressure_df_1 = pd.read_csv(
    'data/20240829_162215_Pd_D_run_start.csv',
    names=['time', 'Voltage1', 'Voltage2', 'Voltage3', 'Voltage4'],
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time',
    header=None
)

pressure_df_2 = pd.read_csv(
    'data/20240830_144123_Pd_D_run_2.csv',
    names=['time', 'Voltage1', 'Voltage2', 'Voltage3', 'Voltage4'],
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time',
    header=None
)

pressure_df = pd.concat([pressure_df_1, pressure_df_2])
```

```python colab={"base_uri": "https://localhost:8080/"} id="5a016413-224d-4ffd-bdd5-1e403830f278" outputId="04d83ba0-4bd8-4604-84b7-dd59ee2a7fd6"
# Print out basic description of the data, including any NaNs
print_info(pressure_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="a2c863e9-5058-4f25-841e-3a1d0f9bb292" outputId="6184bf22-9e90-4592-f2ff-4a46088c59ca"
pressure_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="hmG1mduzeawm" outputId="4c711e43-bbdd-47cf-bb3d-7f256ba1f5a9"
plt.figure(figsize=(8, 4))
plt.plot(pressure_df['Voltage1'])
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {pressure_df.index[0].date()}")
plt.show()
```

<!-- #region id="11c76ccf-9880-4871-9f87-61fb11362e91" -->
## Processing the data
<!-- #endregion -->

<!-- #region id="b9921455-0c57-4950-9a7a-3983f3c467b2" -->
### Corrupt data

There is a large spike in the temperature data about half way through 2024-09-01. We can identify the precise time be seeing where the temperature changes by more than 10C over a single measurement.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 175} id="6b38008a-9cd4-4de3-a3cf-141803421fea" outputId="5b134925-48e9-403d-fa40-b8e078b5848e"
temperature_df[temperature_df['Temperature (C)'].diff().abs() > 10]
```

<!-- #region id="b9122cfc-be52-420c-b49e-aa7947fc8829" -->
Looks like the time that `Thermocouple1Ch1` changed a lot was also the time when `Thermocouple1Ch2` - this suggests an error in the data logging. Let's zoom into this region to see a bit more detail.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="ba18696a-97ae-454f-914a-8e690c903984" outputId="113d3783-1b5d-4278-f439-d8fd59d59037"
temperature_df['Temperature (C)']['2024-09-01 12:12':'2024-09-01 12:14'].plot();
```

<!-- #region id="f34a612f-56ae-4704-888e-e56fe7769fe3" -->
Indeed, it looks like just a single problem point which we can safely remove.
<!-- #endregion -->

```python id="6a2252a4-5dcd-41b4-90f4-94c46ab04628"
# Drop corrupt temperature readings
temperature_df.drop(temperature_df[temperature_df['Temperature (C)'].diff().abs() > 10].index, inplace=True)
```

<!-- #region id="1b1d8485-4770-4116-9bd3-39eefb963bdf" -->
### Noisy data

The pressure data in particular is very noisy. Since we'll be later relying on the initial values of temperature and pressure to calculate the number of deuterium atoms in the chamber at the start of the experiment, it seems sensible to smooth out the pressure data to get a more reliable value.

Although the temperature data is not as noisy, since we need pressure and temperature to calculate the number of deuterium atoms, then it seems sensible to perform the same processing to the temperature data as well.
<!-- #endregion -->

```python id="7abc0669-856f-4e1f-951c-2c49562f3c5a"
# Smoothing with a rolling mean over a 5-point window
pressure_df['Voltage1'] = pressure_df['Voltage1'].rolling(window=200, center=True).mean()
temperature_df['Temperature (C)'] = temperature_df['Temperature (C)'].rolling(window=200, center=True).mean()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="2513e948-38c5-436d-b8a0-5b51f58c1b7a" outputId="7708d700-3f9d-4d0d-9bff-4eab8f0acbb3"
pressure_df['Voltage1'].plot().set_ylabel("Voltage (V)");
```

```python colab={"base_uri": "https://localhost:8080/", "height": 444} id="0089c93d-6545-4dc8-9b3b-e1cd8975dd91" outputId="2766f842-7b14-4434-8200-f223d71e8661"
temperature_df['Temperature (C)'].plot().set_ylabel("Temperature (C)");
```

<!-- #region id="e0364ca9-7d46-4c9d-a743-0d1077f46b45" -->
### Combining data

To derive physical quantities from several diagnostics, we need to have simultaneous measurements. We'll therefore need to do some interpolation of the data. This is going to involve:
1. Mapping all measurements to the nearest second
2. Selecting overlapping time intervals from the data
3. Combining the data from all diagnostics into one dataframe
4. Interpolate to fill the NaNs that result from combining the data in step 3
5. Drop any NaNs that can sometimes be generated at the edges of the time range
<!-- #endregion -->

```python id="e0fa56de-0ec6-4b9c-b2a7-4da02eb09812"
combined_df = process_data([temperature_df, pressure_df,heating_df], meta)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="ASNKpZWVlcFR" outputId="7d109bb5-6c23-4f3a-bb14-1a8ec00a4ae0"
combined_df.head()
```

<!-- #region id="81193f3a-b3e7-4c35-a73f-1707770ae8eb" -->
### Pressure readings
The next processing step involves converting the Voltage measurement into a pressure value.
<!-- #endregion -->

```python id="7e42e0a5-9841-4549-b4ed-e7fed5d7ac3d"
# Constants required to convert pressure gauge voltage into a pressure reading in bar
Resistor = 650  # resistance in Ohms
current_offset_mA = 4  # 4 mA corresponds to 0 bar
pressure_range_bar = 7  # Range is from 0 to 7 bar
current_range_mA = 16  # From 4 mA to 20 mA, so the range is 16 mA

# Calculating current in amperes
pressure_sensor_current = combined_df['Voltage1'] / Resistor

# Calculating pressure in bars
combined_df['Pressure (Bar)'] = 1 + ((pressure_sensor_current * 1000 - current_offset_mA) * pressure_range_bar) / current_range_mA
```

<!-- #region id="b511b1a1-bfb0-4b50-be4c-e61ee4333ce0" -->
### Heating power
The next step is calculating the heating power
<!-- #endregion -->

```python id="d7a75b9f-c782-4a91-809a-9ec2ebc23638"
combined_df['Power (W)'] = combined_df['Voltage']*combined_df['Current']
```

<!-- #region id="ea652753-5d8e-4a87-9d2c-868577ff1b70" -->
### Inferring deuterium loading

The deuterium loading is inferred by using the ideal gas law to calculate how many deuterium molecules are present in the gas over time:

$$N_{D_2} = \frac{PV}{k_B T}$$

and associating any changes $\Delta N_{D_2}$ with deuteium entering the lattice. We can then calculate the loading based on the number of lattice atoms $N_{lattice}$:

$$N_{lattice} = 2.19\times 10^{21}$$

The loading is then:

$$2\frac{\Delta N_{D_2}}{N_{lattice}}$$

The factor 2 arrises because a single $D_2$ molecule becomes 2 deuterons once inside the lattice.
<!-- #endregion -->

```python id="3c4e4339-bd7f-4632-9c8d-21aa23729b88"
# Constants
V = 0.19 / 1000  # Volume of the container in m^3
kB = 1.3806503e-23  # Boltzmann constant in J/K
N_lattice = 2.19e21
```

```python id="64f11bfe-a2d9-45b9-8069-3206860299e2"
combined_df['$D_2$ molecules'] = (combined_df['Pressure (Bar)']*1e5 * V) / (kB * (combined_df['Temperature (C)'] + 273.15))
```

```python id="56c3bcb5-de7a-4a91-8583-3922d6f47f00"
combined_df['D/Pd Loading'] = 2*(combined_df.iloc[0]['$D_2$ molecules'] - combined_df['$D_2$ molecules']) / N_lattice
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="6d44a2b5-ddd6-4024-b281-dbd293f6c85c" outputId="b9c76c13-badf-4bd1-849d-abcf2e01cd81"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['$D_2$ molecules'])
plt.xlabel('Time')
plt.ylabel('$D_2$ molecules')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="6abfec40-461f-46e7-a987-00218be5b56f" outputId="2014fd9a-0499-4342-d111-f3e80a1001bf"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['D/Pd Loading'])
plt.xlabel('Time')
plt.ylabel('D/Pd Loading')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

<!-- #region id="7f7d99e7-0c45-48f4-9783-d82773ebd25f" -->
## Visualising the data
<!-- #endregion -->

<!-- #region id="_6AmD2slooIP" -->
Let's look at the whole data range first
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 805} id="4022b73b-c6d1-4bf8-b8f1-9529f35f23dc" outputId="7f5871dc-9e51-4f9d-8f98-11237d4a846c"
fig, axes = plot_panels(combined_df, ['Power (W)','Temperature (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                        colors=['orange','blue', 'green', 'red'])
```

<!-- #region id="e26bfe5b-772c-48f8-a363-5a770aa5afcd" -->
Let's also take a look at a pressure/temperature phase space plot, for ideal gas behaviour, we'd expect

$$10^{5}P(atm) = \frac{N_{D_2}k_BT(C)}{V} + \frac{273.15N_{D_2}k_B}{V}$$

Because the number of gas molecules $N_{D_2}$ appears in both the gradient term and offset term, the motion around phase space can look quite complicated. Let's see.
<!-- #endregion -->

```python id="fc34463a-1e17-45db-a3e1-daee61c25494" outputId="f06c4724-3d26-4076-ebc7-ac0e41dc16a1"
# Create a scatter plot of pressure vs. temperature
plt.figure(figsize=(6, 6))
plt.scatter(combined_df['Temperature (C)'], combined_df['Pressure (Bar)'], marker=".", color='orange', alpha=0.7)

# Label axes
plt.xlabel('Temperature (C)')
plt.ylabel('Pressure (bar)')

# Add title
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")

# Display the plot
plt.show()
```

<!-- #region id="bbee463c-41c5-4f81-864e-a9c479e0441e" -->
If we down-sample the data a bit, we can get a bit more sense of the dynamics from the scatter plot. It also allows us to see potentially interesting points that we'd like to pay more attention to. For example:
<!-- #endregion -->

```python id="5e5cf7e2-194f-4b33-a2d2-ae574249e6ca" outputId="09ee5513-a125-4c8f-bebd-f6eb4a262f01"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Power (W)','Temperature (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                   "Temperature (C)", "Pressure (Bar)",
                        colors=['orange','blue', 'green', 'red'], downsample=100, marker="2024-08-31 21:00")
```

<!-- #region id="b4b2e961-3bac-4186-9944-3e5487bd4038" -->
And by far the best way to get a feeling for the dynamics is to animate everything.
<!-- #endregion -->

```python id="4c5714c1-3d50-492c-9917-1e4fe372f7a2" outputId="98df3c08-972b-4975-92bd-e5aab2db1fb9"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Power (W)','Temperature (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                  "Temperature (C)", "Pressure (Bar)",
                        colors=['orange','blue', 'green', 'red'], downsample=100, animate=True)
```

```python id="5eb55ab2-4fde-4f5b-a059-769b411e465e" outputId="8ebd7f54-fbd9-4722-85bb-5dfcef074c64"
# If working in colab, then set embed=True
Video("media/Palladium foil 2024-08-29.mp4", embed=False, width=800)
```

<!-- #region id="a618e075-9877-4207-b5c3-27c9bcbdec19" -->
Let's now revisit the P-T phase diagram and this time overlay what we would expect from the ideal gas law.
<!-- #endregion -->

```python id="7c1101f9-0ed9-423a-941f-cfdc98f3154b" outputId="77b3be9c-0a65-44b7-d139-534e54eaa61f"
downsample = 100

# Downsample the data
combined_df_downsampled = combined_df.iloc[::downsample]

# Create a scatter plot of pressure vs. temperature
plt.figure(figsize=(6, 6))
plt.scatter(combined_df_downsampled['Temperature (C)'], combined_df_downsampled['Pressure (Bar)'], marker=".", color='orange', alpha=0.7)

# Label axes
plt.xlabel('Temperature (C)')
plt.ylabel('Pressure (Bar)')

# Add title
plt.title(f"{meta['descriptor']} {combined_df_downsampled.index[0].date()}")

combined_df['$D_2$ molecules'].min()

# Generate ideal gas lines for different N_D2 values
N_D2_values = np.linspace(combined_df['$D_2$ molecules'].max(),
                          combined_df['$D_2$ molecules'].min(), 5)

# # Add custom values, e.g., 100 and 200
# custom_values = [100, 200]
# N_D2_values = np.append(N_D2_values, custom_values)
# N_D2_values = np.unique(N_D2_values)  # Optional: Remove duplicates and sort

temperatures_C = combined_df_downsampled['Temperature (C)']
temperatures_K = temperatures_C + 273.15  # Convert Celsius to Kelvin

for N_D2 in N_D2_values:
    # Calculate pressure in Pa and convert to Bar (1 Pa = 1e-5 Bar)
    pressures_Pa = (N_D2 * kB * temperatures_K) / V
    pressures_Bar = pressures_Pa * 1e-5

    # Extract the mantissa and exponent separately
    mantissa = f"{N_D2:.1e}".split("e")[0]  # Get the coefficient
    exponent = f"{N_D2:.1e}".split("e")[1]  # Get the exponent, like +21 or +22
    exponent = exponent.replace('+', '')    # Remove any plus sign from the exponent

    # Create the label with explicit LaTeX formatting
    label = f'$N_{{D_2}} = {mantissa} \\times 10^{{{exponent}}}$'

    # Plot the line for this N_D2 value
    plt.plot(temperatures_C, pressures_Bar, label=label)

# Show legend for clarity
plt.legend(title='Ideal Gas Lines')

# Display the plot
plt.show()
```

```python id="67ad6c06-13e7-405a-847e-fbf50d1c26ce"

```
