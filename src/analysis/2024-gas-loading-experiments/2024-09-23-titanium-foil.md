---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="3f826b5b-1cf7-45b2-9622-89c10dbf1eb2" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/2024-gas-loading-experiments/2024-09-23-titanium-foil.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/2024-gas-loading-experiments/2024-09-23-titanium-foil.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
# 2024-09-23 Titanium foil (etched)
<!-- #endregion -->

<!-- #region id="487e78f6-0666-4d0c-ade0-30403aa31975" -->
A 310mg etched titanium foil is gas loaded with deuterium, in a 0.19L chamber.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6e5640a1-12da-4157-a5e8-5f73f882e6a7" outputId="79218982-13bd-429f-b370-a456b324db89"
# RUN THIS TO MAKE SURE WE CAN IMPORT LIBS WHETHER WE ARE IN COLAB OR LOCAL

import sys
import os

# Check if running in Google Colab
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !git clone https://github.com/project-ida/arpa-e-experiments.git
    sys.path.insert(0, '/content/arpa-e-experiments')
else:
    # Running locally
    # Get the parent directory (two levels up from the current directory)
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    # Add the parent directory to sys.path
    sys.path.insert(0, project_root)
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
# - load_data
from libs.helpers import *

# Necessary for using load_data on password protected data urls
# - authenticate
# - get_credentials
from libs.auth import *
```

```python id="77ff6961-702c-4e53-b339-6d98a95d73c3"
meta = {
    "descriptor" : "Etched titanium foil" # This will go into the title of all plots
}
```

<!-- #region id="d1d7c4fc-7df2-4c54-8be1-2750a9071260" -->
## Reading the raw data
<!-- #endregion -->

<!-- #region id="81tl_imIb5oB" -->
### Temperature
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fde663ef-7691-4c50-8a21-df4e77c67d25" outputId="b7a70965-445b-4fe5-b771-31771e166a8a"
# Read the tempearture data
temperature_df = load_data('http://nucleonics.mit.edu/data/csv-files/loading%20deloading%20runs/thermocouples_september_ti/thermocouples_september_ti-1-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="91c572f5-e03c-470d-961f-4e86959c0ed6" outputId="60629811-2b16-4043-a4ed-be480673c364"
# Print out basic description of the data, including any NaNs
print_info(temperature_df)
```

<!-- #region id="I5nT2hx4lFdd" -->
`Thermocouple1Ch4` was offline for the whole experiment so we'll drop it.

`Thermocouple1Ch3` is the room temperature measurement and had a small number of issues. We'll fix this during the processing stage.
<!-- #endregion -->

```python id="soi2JWUplH5O"
temperature_df.drop('Thermocouple1Ch4', axis=1, inplace=True) # Drop Thermocouple1Ch4
```

<!-- #region id="58c17aeb-f5d2-4560-9f54-f403193bf342" -->
Since we'll only be interested in `Thermocouple1Ch1`, we'll rename it to make plotting a bit easier later.
<!-- #endregion -->

```python id="0d74f788-b490-4ae5-9164-07bb13469dec"
temperature_df.rename(columns={'Thermocouple1Ch1': 'Temperature (C)'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="e3fd770c-4082-432d-85c6-676c0ffdb901" outputId="454ab096-9bdb-4f35-800c-f8c413f23bc0"
plt.figure(figsize=(8, 4))
plt.plot(temperature_df['Temperature (C)'])
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {temperature_df.index[0].date()}")
plt.show()
```

<!-- #region id="KwGr1eAUcY_h" -->
### Pressure
<!-- #endregion -->

```python id="D8kU4YY7b3W7"
# Read the pressure data
pressure_df = pd.read_csv(
    'data/20240923_192738_Ti_etched_run_2_cycles+RTeq.csv',
    names=['time', 'Voltage1', 'Voltage2', 'Voltage3', 'Voltage4'],
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time',
    header=None
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="8e40efa6-0ade-4172-a76c-ae404ad8fd26" outputId="863c0454-4604-45bd-9a93-11068f927f49"
# Print out basic description of the data, including any NaNs
print_info(pressure_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="hmG1mduzeawm" outputId="6247f2f9-c848-484b-cb1c-11eb19d49ee2"
plt.figure(figsize=(8, 4))
plt.plot(pressure_df['Voltage1'])
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {pressure_df.index[0].date()}")
plt.show()
```

<!-- #region id="23a119e2-30dc-4791-8550-c6fd3d81d573" -->
## Processing the data
<!-- #endregion -->

<!-- #region id="a9a15a70-b808-45f4-b55b-fb2b4a693836" -->
### Corrupt data
<!-- #endregion -->

<!-- #region id="X_U8NgBXf_5e" -->
There is a large spike in the pressure data about half way through 2024-09-27. We can identify the precise time be seeing where the voltage changes by more than 0.5V over a single measurement.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 143} id="QSYtI8O8gQq1" outputId="f692ed66-9571-495b-8844-dddf08d6ccd8"
pressure_df[pressure_df['Voltage1'].diff().abs() > 0.5]
```

<!-- #region id="qMEHIHTjgl_m" -->
Let's zoom in to this region
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="9lP2pNOwfrpE" outputId="e9814e80-68c7-4393-e9d9-0e5dfe839eac"
pressure_df['Voltage1']['2024-09-27 16:10':'2024-09-27 16:13'].plot();
```

<!-- #region id="Fut6DLVmhW-1" -->
It's likely that the largest spike is a result of data logging issues and so it seems sensible to remove the data points.
<!-- #endregion -->

```python id="HXe07uAte95a"
# Drop corrupt pressure readings
pressure_df.drop(pressure_df[pressure_df['Voltage1'].diff().abs() > 0.5].index, inplace=True)
```

<!-- #region id="cfdbdd56-ad3e-4a51-ad5c-eb99307de6de" -->
### Combining data
<!-- #endregion -->

<!-- #region id="e0364ca9-7d46-4c9d-a743-0d1077f46b45" jp-MarkdownHeadingCollapsed=true -->
To derive physical quantities from several diagnostics, we need to have simultaneous measurements. We'll therefore need to do some interpolation of the data. This is going to involve:
1. Mapping all measurements to the nearest second
2. Selecting overlapping time intervals from the data
3. Combining the data from all diagnostics into one dataframe
4. Interpolate to fill the NaNs that result from combining the data in step 3
5. Drop any NaNs that can sometimes be generated at the edges of the time range
<!-- #endregion -->

```python id="e0fa56de-0ec6-4b9c-b2a7-4da02eb09812"
combined_df = process_data([temperature_df, pressure_df], meta)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="ASNKpZWVlcFR" outputId="9daa9eed-63bd-4fc2-d0c5-208de2da3264"
combined_df.head()
```

<!-- #region id="5a570c69-908c-4530-a30d-1f0a67b5c60e" -->
### Pressure readings
The next processing step involves converting the Voltage measurement into a pressure value.
<!-- #endregion -->

```python id="a68f9cd1-e486-4c50-bb93-30feff9f1076"
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

<!-- #region id="QYN66zEKltTz" -->
### Inferring deuterium loading

The deuterium loading is inferred by using the ideal gas law to calculate how many deuterium molecules are present in the gas over time:

$$N_{D_2} = \frac{PV}{k_B T}$$

and associating any changes $\Delta N_{D_2}$ with deuteium entering the lattice. We can then calculate the loading based on the number of lattice atoms $N_{lattice}$:

$$N_{lattice} = \frac{m_{sample}}{m_{Ti}}$$

$$N_{lattice} = \frac{310\times 10^{-6}}{7.949\times 10^{-26}}$$

The loading is then:

$$2\frac{\Delta N_{D_2}}{N_{lattice}}$$

The factor 2 arrises because a single $D_2$ molecule becomes 2 deuterons once inside the lattice.
<!-- #endregion -->

```python id="d82GrK71lpL1"
# Constants
V = 0.19 / 1000  # Volume of the container in m^3
kB = 1.3806503e-23  # Boltzmann constant in J/K
N_lattice = 310e-6 / 7.949e-26
```

```python id="U2JLGyxDl6XY"
combined_df['$D_2$ molecules'] = (combined_df['Pressure (Bar)']*1e5 * V) / (kB * (combined_df['Temperature (C)'] + 273.15))
```

```python id="pC6lX3Rml8oG"
combined_df['D/Ti Loading'] = 2*(combined_df.iloc[0]['$D_2$ molecules'] - combined_df['$D_2$ molecules']) / N_lattice
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="ef7e3240-5e28-41b1-a58f-fdbe701e949d" outputId="2d79084f-3406-455a-96e3-f9aceae028ca"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['$D_2$ molecules'])
plt.xlabel('Time')
plt.ylabel('$D_2$ molecules')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="DJozVX8Gl-0e" outputId="aa11e4f0-ba39-43d4-abdc-7ec04db733b5"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['D/Ti Loading'])
plt.xlabel('Time')
plt.ylabel('D/Ti Loading')
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

```python colab={"base_uri": "https://localhost:8080/", "height": 806} id="4022b73b-c6d1-4bf8-b8f1-9529f35f23dc" outputId="7c8e79c4-fea0-4a8b-bad0-c97dd3ddb720"
fig, axes = plot_panels(combined_df, ['Temperature (C)', 'Pressure (Bar)', 'D/Ti Loading'],
                        colors=['blue', 'green', 'red'])
```

<!-- #region id="612a1eb0-4e71-4303-b099-dd3c9dc0dbe2" -->
Let's also take a look at a pressure/temperature phase space plot, for ideal gas behaviour, we'd expect

$$10^{5}P(atm) = \frac{N_{D_2}k_BT(C)}{V} + \frac{273.15N_{D_2}k_B}{V}$$

Because the number of gas molecules $N_{D_2}$ appears in both the gradient term and offset term, the motion around phase space can look quite complicated. Let's see.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="39ff5bb2-ccf7-4975-8bfd-9245dfb77a5e" outputId="21243c10-c492-46c4-8a7e-d16f46119303"
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

<!-- #region id="431c6bed-519c-4077-bccd-cf69aa0cbac4" -->
If we down-sample the data a bit, we can get a bit more sense of the dynamics from the scatter plot. It also allows us to see potentially interesting points that we'd like to pay more attention to. For example:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 710} id="bc589978-a18c-4300-ac33-74c835cedba1" outputId="d48cce6d-36a0-4d7a-a00e-c222cf8f4115"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Temperature (C)', 'Pressure (Bar)', 'D/Ti Loading'],
                   "Temperature (C)", "Pressure (Bar)",
                        colors=['blue', 'green', 'red'], downsample=100, marker="2024-09-26 12:00")
```

<!-- #region id="8e60da6e-2b92-4ae6-9610-613480dac622" -->
And by far the best way to get a feeling for the dynamics is to animate everything.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="c5e307f8-ce51-413c-b474-870a6dbbd123" outputId="a3bb6213-35a8-46f3-b84e-7a22431ffd7c"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Temperature (C)', 'Pressure (Bar)', 'D/Ti Loading'],
                  "Temperature (C)", "Pressure (Bar)",
                        colors=['blue', 'green', 'red'], downsample=100, animate=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 421, "resources": {"http://localhost:8080/media/Etched%20titanium%20foil%202024-09-23.mp4": {"data": "", "headers": [["content-length", "0"]], "ok": false, "status": 404, "status_text": ""}}} id="98c4743e-ec8c-4d53-ac93-dd5685e756eb" outputId="3fd831ef-9061-4f88-c00e-e0299bca9fb8"
# If working in colab, then set embed=True
Video("media/Etched titanium foil 2024-09-23.mp4", embed=False, width=800)
```

<!-- #region id="5163525b-21b4-4745-b09c-70bb28a5e056" -->
Let's now revisit the P-T phase diagram and this time overlay what we would expect from the ideal gas law.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="718630ac-53ff-4873-a41b-e5ad228c686e" outputId="eb955fbe-963c-44f9-bf0b-4bd17fd3996c"
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
N_D2_values = np.linspace(9.2e21, 5.85e21, 5)

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

```python id="204b50a0-a5f2-46e8-8aff-ced78143355d"

```
