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
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/2024-gas-loading-experiments/2024-10-05-titanium-foil.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/2024-gas-loading-experiments/2024-10-05-titanium-foil.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
# 2024-10-05 Titanium foil
<!-- #endregion -->

<!-- #region id="487e78f6-0666-4d0c-ade0-30403aa31975" -->
A 310mg titanium foil is gas loaded with deuterium, in a 0.19L chamber.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6e5640a1-12da-4157-a5e8-5f73f882e6a7" outputId="fe3e8090-2b2e-420f-cec4-cb920fb7d169"
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

```python id="e2ed811c-a621-4cb3-befb-2e31eafa6a78"
meta = {
    "descriptor" : "Titanium foil" # This will go into the title of all plots
}
```

<!-- #region id="d1d7c4fc-7df2-4c54-8be1-2750a9071260" -->
## Reading the raw data
<!-- #endregion -->

<!-- #region id="81tl_imIb5oB" -->
### Temperature
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fde663ef-7691-4c50-8a21-df4e77c67d25" outputId="9ea75149-cb7e-45fe-c2c7-d9a405ee38eb"
# Read the tempearture data
temperature_df = load_data('http://nucleonics.mit.edu/data/csv-files/loading%20deloading%20runs/thermocouples_october_ti/thermocouples_october_ti-1-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="59034ea9-1c88-403e-92d8-d2591db3592c" outputId="d78dc8b3-ea41-40d7-ff4f-1aabb1bfa0db"
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

<!-- #region id="778a994b-43a7-4c2d-85ed-6e85e36c85a4" -->
Since we'll only be interested in `Thermocouple1Ch1`, we'll rename it to make plotting a bit easier later.
<!-- #endregion -->

```python id="0e92ea8c-dbe2-4c13-b04b-aa116da6086b"
temperature_df.rename(columns={'Thermocouple1Ch1': 'Temperature (C)'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="e3fd770c-4082-432d-85c6-676c0ffdb901" outputId="5ae59cd9-589f-40cb-ac8f-062313fc18f4"
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

```python colab={"base_uri": "https://localhost:8080/"} id="D8kU4YY7b3W7" outputId="f1234c72-8a76-4a3d-fb86-202f4fcc2012"
# Read the pressure data
pressure_df = load_data('http://nucleonics.mit.edu/data/csv-files/loading%20deloading%20runs/thermocouples_october_ti/thermocouples_october_ti-3-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="af60e3d7-edb5-4d7c-a9aa-bdf14cdb9350" outputId="f8ae22e3-ddc7-4be8-d56e-47d4c3cff2ee"
# Print out basic description of the data, including any NaNs
print_info(pressure_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="hmG1mduzeawm" outputId="7b1af1b2-164f-4e06-a866-c9f828959419"
plt.figure(figsize=(8, 4))
plt.plot(pressure_df['Voltage1'])
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {pressure_df.index[0].date()}")
plt.show()
```

<!-- #region id="e0364ca9-7d46-4c9d-a743-0d1077f46b45" -->
## Processing the data

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

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="ASNKpZWVlcFR" outputId="f1475c5f-ae36-4486-c5c9-b5056d502e4c"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="31d3ceb7-24a6-450e-8407-960a83072739" outputId="7e473396-2ad4-488c-fac8-ca6b9e6b99e8"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['$D_2$ molecules'])
plt.xlabel('Time')
plt.ylabel('$D_2$ molecules')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="DJozVX8Gl-0e" outputId="613eb439-cf21-4a6e-e7e1-74ecd7f981a0"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 805} id="4022b73b-c6d1-4bf8-b8f1-9529f35f23dc" outputId="ad6a3295-b2ae-4358-d1fb-7680e560e13f"
fig, axes = plot_panels(combined_df, ['Temperature (C)', 'Pressure (Bar)', 'D/Ti Loading'],
                        colors=['blue', 'green', 'red'])
```

<!-- #region id="d9-KwPgVqlM8" -->
Next, we'll take a more zoomed in view.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 806} id="c1b5ac37-b92f-43e9-a357-82e36864555e" outputId="65b58867-1e19-4c9d-bbf7-cfccdf282c12"
fig, axes = plot_panels(combined_df, ['Temperature (C)', 'Pressure (Bar)', 'D/Ti Loading'],
                        colors=['blue', 'green', 'red'], start="2024-10-05 18:00", stop="2024-10-06 12:00")
```

<!-- #region id="ed17158d-5c7b-4d0f-91c4-b88b077ea9a9" -->
Let's also take a look at a pressure/temperature phase space plot, for ideal gas behaviour, we'd expect

$$10^{5}P(atm) = \frac{N_{D_2}k_BT(C)}{V} + \frac{273.15N_{D_2}k_B}{V}$$

Because the number of gas molecules $N_{D_2}$ appears in both the gradient term and offset term, the motion around phase space can look quite complicated. Let's see.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="076346ab-f26e-4831-a005-aa5312e7c276" outputId="5c0dcca0-fbca-4a7c-e63d-0a68372224fd"
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

<!-- #region id="7114d501-08f1-432f-b64a-184bad859353" -->
If we down-sample the data a bit, we can get a bit more sense of the dynamics from the scatter plot. It also allows us to see potentially interesting points that we'd like to pay more attention to. For example:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 805} id="732c2df7-5f64-4341-8ffc-682e45863fe6" outputId="0ae6938d-ebeb-4719-9a7b-13ef3a5d8fba"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Temperature (C)', 'Pressure (Bar)', 'D/Ti Loading'],
                   "Temperature (C)", "Pressure (Bar)",
                        colors=['blue', 'green', 'red'], downsample=100, marker="2024-10-06 00:00")
```

<!-- #region id="a2fd496a-061c-49de-a800-d5747d1394c6" -->
And by far the best way to get a feeling for the dynamics is to animate everything.
<!-- #endregion -->

```python id="61f37fcd-4427-4d76-9704-c20509add038" outputId="13fbd607-7012-4581-fac1-9d1fe1bb0f13"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Temperature (C)', 'Pressure (Bar)', 'D/Ti Loading'],
                   "Temperature (C)", "Pressure (Bar)",
                        colors=['blue', 'green', 'red'], downsample=100,  animate=True)
```

```python id="02873952-b88a-427c-b40d-06d5d5c8e3e6" outputId="7e204c34-44a4-4f57-fb40-f1484d8ab019"
# If working in colab, then set embed=True
Video("media/Titanium foil 2024-10-05.mp4", embed=False, width=800)
```

<!-- #region id="0bb6345d-50a4-45ca-8166-e04f82eff3aa" -->
Let's now revisit the P-T phase diagram and this time overlay what we would expect from the ideal gas law.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="7356845f-f97c-4eef-88f9-64bdd2f29025" outputId="72e09f6e-7c38-4ce1-9cd2-fb5fe24bca7d"
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
N_D2_values = np.linspace(combined_df['$D_2$ molecules'].max(), 6.2e21, 5)

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

```python id="847adaa2-a467-4562-8137-6a154a990e1a"

```
