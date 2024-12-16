---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="3f826b5b-1cf7-45b2-9622-89c10dbf1eb2" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/2024-09-16-palladium-foil.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/2024-09-16-palladium-foil.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
# 2024-09-16 Palladium foil (annealed)
<!-- #endregion -->

<!-- #region id="487e78f6-0666-4d0c-ade0-30403aa31975" -->
A xy mg annealed palladium foil is gas loaded with deuterium, in a 0.19L chamber.
<!-- #endregion -->

```python id="6e5640a1-12da-4157-a5e8-5f73f882e6a7"
# RUN THIS IF YOU ARE USING GOOGLE COLAB
import sys
import os
!git clone https://github.com/project-ida/arpa-e-experiments.git
sys.path.insert(0,'/content/arpa-e-experiments')
os.chdir('/content/arpa-e-experiments')
```

```python id="a9b070cf-0f22-4946-a040-1860350240d4" outputId="c562377b-1c11-47ae-f6bb-8edde980edfe"
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

```python id="01ba4ae4-0842-4e70-98e0-655cd02493c0"
meta = {
    "descriptor" : "Annealed palladium foil" # This will go into the title of all plots
}
```

<!-- #region id="d1d7c4fc-7df2-4c54-8be1-2750a9071260" -->
## Reading the raw data
<!-- #endregion -->

<!-- #region id="81tl_imIb5oB" -->
### Temperature
<!-- #endregion -->

```python id="fde663ef-7691-4c50-8a21-df4e77c67d25"
# Read the temperature data
temperature_df = pd.read_csv(
    'data/20240916_201028_Pd_annealed_run_4_cycles_temperatures_extratime.csv',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)
```

```python id="1e7da5f2-faf5-4e59-a13e-96b3d0a43f67" outputId="372e4a89-1fa5-4e39-c822-999409558b01"
# Print out basic description of the data, including any NaNs
print_info(temperature_df)
```

<!-- #region id="I5nT2hx4lFdd" -->
`Thermocouple1Ch4` was offline for the whole experiment so we'll drop it.

`Thermocouple1Ch2` and `Thermocouple1Ch3` have a few NaN which we'll fix this during the processing stage.
<!-- #endregion -->

```python id="soi2JWUplH5O"
temperature_df.drop('Thermocouple1Ch4', axis=1, inplace=True) # Drop Thermocouple1Ch4
```

<!-- #region id="3b8e9a2d-4a36-4a00-9d73-1dd60f76bd6a" -->
Since we'll only be interested in `Thermocouple1Ch1`, we'll rename it to make plotting a bit easier later.
<!-- #endregion -->

```python id="de5a1a92-8f76-49e0-a2ba-cd6edba7bbd7"
temperature_df.rename(columns={'Thermocouple1Ch1': 'Temperature (C)'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="e3fd770c-4082-432d-85c6-676c0ffdb901" outputId="bc14f813-8466-4926-ccb8-90cf71990347"
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
    'data/20240916_201028_Pd_annealed_run_4_cycles_powersupply_extratime.csv',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)
```

```python id="8b2fd774-c87b-460d-9d88-10dc161280e1" outputId="6a344aa6-8c53-463c-9171-3da786787686"
# Print out basic description of the data, including any NaNs
print_info(heating_df)
```

```python id="f72ac15c-1a1f-4855-b3ea-eff6979553ce" outputId="63752c09-5505-4e3c-9638-a5ccab201b13"
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
pressure_df = pd.read_csv(
    'data/20240916_201028_Pd_annealed_run_4_cycles-plus.csv',
    names=['time', 'Voltage1', 'Voltage2', 'Voltage3', 'Voltage4'],
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time',
    header=None
)
```

```python id="e8de5175-490e-4e32-9a74-d50ddac4d697" outputId="e1a79c60-7c57-4d79-9c5e-4d7f6b396377"
# Print out basic description of the data, including any NaNs
print_info(pressure_df)
```

```python id="a2c863e9-5058-4f25-841e-3a1d0f9bb292" outputId="62a0a92a-2504-4dda-9265-08e8aa1355dd"
pressure_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="hmG1mduzeawm" outputId="ba41ed0f-298e-4caf-cbdf-fd8c161e1fe6"
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

There is a large spike in the pressure data at the start of the experiment. We can identify the precise time be seeing where the voltage changes by more than 0.5 over a single measurement.
<!-- #endregion -->

```python id="6b38008a-9cd4-4de3-a3cf-141803421fea" outputId="104e41f3-ff63-435d-b785-e49bafb86a9b"
pressure_df[pressure_df['Voltage1'].diff().abs() > 0.5]
```

<!-- #region id="b9122cfc-be52-420c-b49e-aa7947fc8829" -->
Let's zoom in to this region
<!-- #endregion -->

```python id="ba18696a-97ae-454f-914a-8e690c903984" outputId="47acc406-9965-437f-bb07-8e3d404a165b"
pressure_df['Voltage1']['2024-09-16 20:11':'2024-09-16 20:17'].plot();
```

<!-- #region id="f34a612f-56ae-4704-888e-e56fe7769fe3" -->
This looks less like corrupt data and more like when the chamber was suddenly filled with gas at the start of the experiment. We'll therefore keep this data for now, but we'll want to modify the range of data to exclude the pre-fill stage later on.
<!-- #endregion -->

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

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="ASNKpZWVlcFR" outputId="67d7be78-cb7c-40d1-f899-1cdb7dcf2dd2"
combined_df.head()
```

<!-- #region id="b76f81ed-3b0a-4fae-9785-a4d124bea2b2" -->
We'll now exclude the pre-fill stage to make loading analysis more straightforward later.
<!-- #endregion -->

```python id="abfaa43e-ddaf-47e2-8c7a-3f049c555b05"
combined_df = combined_df['2024-09-16 20:17:00':]
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

```python id="443baf90-8a57-45ae-836d-2d5e6a724f76"
combined_df['D/Pd Loading'] = 2*(combined_df.iloc[0]['$D_2$ molecules'] - combined_df['$D_2$ molecules']) / N_lattice
```

```python id="322aadad-3020-4e79-b8ff-bf775af936e1" outputId="f114694a-1362-4bb8-fd92-4dd5be76764a"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['$D_2$ molecules'])
plt.xlabel('Time')
plt.ylabel('$D_2$ molecules')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

```python id="dd5fb4a8-31ee-4887-bf2d-353ff3673611" outputId="5b222061-b6ea-4864-9d1b-e8568af8f208"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 806} id="4022b73b-c6d1-4bf8-b8f1-9529f35f23dc" outputId="eab1587b-f3fa-4d69-c944-8e4333780a46"
fig, axes = plot_panels(combined_df, ['Power (W)','Temperature (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                        colors=['orange','blue', 'green', 'red'])
```

<!-- #region id="a579eb55-83cc-418b-aaed-34940f08b9ac" -->
Let's also take a look at a pressure/temperature phase space plot, for ideal gas behaviour, we'd expect

$$10^{5}P(atm) = \frac{N_{D_2}k_BT(C)}{V} + \frac{273.15N_{D_2}k_B}{V}$$

Because the number of gas molecules $N_{D_2}$ appears in both the gradient term and offset term, the motion around phase space can look quite complicated. Let's see.
<!-- #endregion -->

```python id="d79346ee-12f3-45eb-81b2-6a28e1e472e2" outputId="324c42ab-e892-40ee-a181-0fae5799d557"
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

<!-- #region id="b0050afa-cd45-4e54-88e2-40c83d38fb03" -->
If we down-sample the data a bit, we can get a bit more sense of the dynamics from the scatter plot. It also allows us to see potentially interesting points that we'd like to pay more attention to. For example:
<!-- #endregion -->

```python id="f687b3d5-44f4-4ae7-a79a-578be3dccd66" outputId="bb46fa98-c2be-4b83-86d5-e3ef2b0ab382"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Power (W)','Temperature (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                   "Temperature (C)", "Pressure (Bar)",
                        colors=['orange','blue', 'green', 'red'], downsample=100, marker="2024-09-19 00:00")
```

<!-- #region id="93f6ce56-f22c-4b6a-9efa-4fa3cbe03a17" -->
And by far the best way to get a feeling for the dynamics is to animate everything.
<!-- #endregion -->

```python id="0ec7e5e3-7ef0-4d68-aba5-1a9676699587" outputId="8cf41e32-2b18-46c2-8524-91332af639e6"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Power (W)','Temperature (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                  "Temperature (C)", "Pressure (Bar)",
                        colors=['orange','blue', 'green', 'red'], downsample=100, animate=True)
```

```python id="b2c74384-5362-49eb-98ea-282cae878d55" outputId="115c38c5-f54c-4111-e79a-4a75405e38c5"
# If working in colab, then set embed=True
Video("media/Annealed palladium foil 2024-09-16.mp4", embed=False, width=800)
```

<!-- #region id="c41c3bff-df8d-4370-ba99-b207937ed69b" -->
Let's now revisit the P-T phase diagram and this time overlay what we would expect from the ideal gas law.
<!-- #endregion -->

```python id="c510b0b8-c20d-4d3e-a31d-e74a41517849" outputId="d1f07d1b-edd6-4274-fe73-5194d0989e4e"
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

# Generate ideal gas lines for different N_D2 values
N_D2_values = np.linspace(1e22, 8e21, 5)

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

```python id="e24dc929-9687-47c8-a860-d892af5a12eb"

```
