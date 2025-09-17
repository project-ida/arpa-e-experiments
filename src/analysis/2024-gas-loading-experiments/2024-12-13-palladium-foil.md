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
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/analysis/2024-gas-loading-experiments/2024-12-13-palladium-foil.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/analysis/2024-gas-loading-experiments/2024-12-13-palladium-foil.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
# 2024-12-13 Palladium foil
<!-- #endregion -->

<!-- #region id="487e78f6-0666-4d0c-ade0-30403aa31975" -->
A 136 mg Palladium foil is gas loaded with deuterium, in a 0.59L chamber.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6e5640a1-12da-4157-a5e8-5f73f882e6a7" outputId="aeceab32-afae-4d4d-9a13-6d6ce3775c05"
# Makes Libs accessible and runs notebook from same location regardless of whether colab or local
!pip install colocal -q
import colocal
root, branch, cwd = colocal.setup()
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

```python colab={"base_uri": "https://localhost:8080/"} id="fde663ef-7691-4c50-8a21-df4e77c67d25" outputId="a4927bc1-943a-4cf7-8cad-e955b3befa5d"
# Read the tempearture data
temperature_df = load_data('http://nucleonics.mit.edu/data/csv-files/loading%20deloading%20runs/new-chamber-first-pd/new-chamber-first-pd-5-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="686ac467-0ef2-48a8-9598-a80f357130a8" outputId="7f1414dc-75fb-4b91-8a10-9cc552de4d1a"
# Print out basic description of the data, including any NaNs
print_info(temperature_df)
```

<!-- #region id="6e7a2688-aa5a-4533-a75e-f86815179b75" -->
We're interested in using `chamber-RTD1` and `chamber-RTD2` to get a sense of the tempreature near the sample. Let's see how close they are to one another.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="xW-mQwpYrzIn" outputId="d7d90428-2894-4ba4-9247-4e11dcdbfe84"
plt.figure(figsize=(8, 4))
plt.plot(temperature_df['chamber-RTD1'], label="RTD1")
plt.plot(temperature_df['chamber-RTD2'], label="RTD2")
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} 2024-12-13")
plt.legend()
plt.show()
```

<!-- #region id="AZliuR7-sVtx" -->
We'll take an average of the RTD's and call it `Temperature (C)`
<!-- #endregion -->

```python id="Unh24H_SsU3Q"
temperature_df['Temperature (C)'] = (temperature_df['chamber-RTD1'] + temperature_df['chamber-RTD2']) / 2
```

<!-- #region id="h85O1nfCwzrt" -->
We're heating the chamber with heating tape. The tape wraps most of the chamber but not all. It's therefore likely that we won't have isothermal conditions. The temperature measurements are taking in the center. They are not going to be a good measure of the average temperature. We need to apply some correction.

As a first attempt, we'll scale $T-T_{ambient}$.
<!-- #endregion -->

```python id="3IlKsnbaxmfj"
T_scale_factor = 0.9
```

```python id="gSobJYJlxtGF"
temperature_df['$\Delta$ T (C)']  = temperature_df['Temperature (C)'] - temperature_df['Temperature (C)'].iloc[0]
```

```python id="mZxCwWSLx0D9"
temperature_df['Inferred mean T (C)'] = temperature_df['Temperature (C)'].iloc[0] + temperature_df['$\Delta$ T (C)']*T_scale_factor
```

<!-- #region id="75744b06-6ec3-49cc-a1db-8d7b075b176f" -->
### Heating power
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3a6692b1-462c-43de-83de-d31864a847b3" outputId="fbe701cf-4f51-4d27-aee3-fff00e5fe052"
# Read the heating power data
heating_df = load_data('http://nucleonics.mit.edu/data/csv-files/loading%20deloading%20runs/new-chamber-first-pd/new-chamber-first-pd-1-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="81174345-4f9b-4095-ab2e-3dfd172ea8a2" outputId="bae08997-892b-4e7b-d1b5-3cac8cc1b494"
# Print out basic description of the data, including any NaNs
print_info(heating_df)
```

<!-- #region id="zlFIC6oEFybb" -->
We'll rename "Output power" to just "Power" convenience in plotting
<!-- #endregion -->

```python id="_tu83TMFFpu6"
heating_df.rename(columns={'Output power (W)': 'Power (W)'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="f72ac15c-1a1f-4855-b3ea-eff6979553ce" outputId="d03fedf0-3953-4838-d064-80909472607a"
plt.figure(figsize=(8, 4))
plt.plot(heating_df['Power (W)'], label="Power (W)")
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} 2024-12-13")
plt.show()
```

<!-- #region id="KwGr1eAUcY_h" -->
### Pressure
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="D8kU4YY7b3W7" outputId="6e1a4fa8-19e9-41f3-d192-c9a6c7a73f47"
# Read the pressure data
pressure_df = load_data('http://nucleonics.mit.edu/data/csv-files/loading%20deloading%20runs/new-chamber-first-pd/new-chamber-first-pd-3-fullres.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="5a016413-224d-4ffd-bdd5-1e403830f278" outputId="b134888e-69e9-4e61-a827-9d4bc7cd7594"
# Print out basic description of the data, including any NaNs
print_info(pressure_df)
```

<!-- #region id="rbdWZx4qxkUl" -->
We'll rename the pressure and remove the Ch3 for convenience in plotting.
<!-- #endregion -->

```python id="4dhrl8p8xjW9"
pressure_df.rename(columns={'Pressure Ch3 (bar)': 'Pressure (Bar)'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="hmG1mduzeawm" outputId="c6b1e8fd-fc90-4eaf-d34c-ebc5a2659965"
plt.figure(figsize=(8, 4))
plt.plot(pressure_df['Pressure (Bar)'])
plt.xlabel('Time')
plt.ylabel('Pressure (bar)')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} 2024-12-13")
plt.show()
```

<!-- #region id="11c76ccf-9880-4871-9f87-61fb11362e91" -->
## Processing the data
<!-- #endregion -->

<!-- #region id="b9921455-0c57-4950-9a7a-3983f3c467b2" -->
### Corrupt data

There looks like there is some spikes in the data near start and end of the experiment but these are not important because the true start and end of the experiment are around 2024-12-14 and 2024-12-20.
<!-- #endregion -->

<!-- #region id="jIFug-smLgEy" -->
When we combine the data, we'll cut off the earlier and later times so that we can more easily calculate the deuterium loading.
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

```python colab={"base_uri": "https://localhost:8080/", "height": 446} id="ASNKpZWVlcFR" outputId="64821e3c-e30a-48d2-bf9e-53d0e783cc7e"
combined_df.head()
```

```python id="OMJWLq_rLy9I"
# Only use times when the experiment was being performed
combined_df = combined_df['2024-12-13 18:32':'2024-12-20 00:00']
```

<!-- #region id="ea652753-5d8e-4a87-9d2c-868577ff1b70" -->
### Inferring deuterium loading

The deuterium loading is inferred by using the ideal gas law to calculate how many deuterium molecules are present in the gas over time:

$$N_{D_2} = \frac{PV}{k_B T}$$

and associating any changes $\Delta N_{D_2}$ with deuteium entering the lattice. We can then calculate the loading based on the number of lattice atoms $N_{lattice}$:

$$N_{lattice} = \frac{m_{sample}}{m_{Pd}}$$

$$N_{lattice} = \frac{120\times 10^{-6}}{1.77\times 10^{-25}}$$

The loading is then:

$$2\frac{\Delta N_{D_2}}{N_{lattice}}$$

The factor 2 arrises because a single $D_2$ molecule becomes 2 deuterons once inside the lattice.
<!-- #endregion -->

```python id="3c4e4339-bd7f-4632-9c8d-21aa23729b88"
# Constants
V = 0.59*0.7 / 1000  # Volume of the container in m^3 taking into account space lost to equipment inside
kB = 1.3806503e-23  # Boltzmann constant in J/K
N_lattice = 136e-6 / 1.77e-25
```

```python id="64f11bfe-a2d9-45b9-8069-3206860299e2"
combined_df['$D_2$ molecules'] = ((combined_df['Pressure (Bar)'])*1e5 * V) / (kB * (combined_df['Inferred mean T (C)'] + 273.15))
```

```python id="56c3bcb5-de7a-4a91-8583-3922d6f47f00"
combined_df['D/Pd Loading'] = 2*(combined_df.iloc[0]['$D_2$ molecules'] - combined_df['$D_2$ molecules']) / N_lattice
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="6d44a2b5-ddd6-4024-b281-dbd293f6c85c" outputId="92fecd5c-7662-4a6c-bfd0-a8a636b1c69a"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['$D_2$ molecules'])
plt.xlabel('Time')
plt.ylabel('$D_2$ molecules')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="6abfec40-461f-46e7-a987-00218be5b56f" outputId="415ea24c-d14f-4a9c-bb4e-53c0dadad78e"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 805} id="4022b73b-c6d1-4bf8-b8f1-9529f35f23dc" outputId="27b67e2f-4d9d-4670-bb94-e1e43e81ff1c"
fig, axes = plot_panels(combined_df, ['Power (W)','Inferred mean T (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                        colors=['orange','blue', 'green', 'red'])
```

<!-- #region id="e26bfe5b-772c-48f8-a363-5a770aa5afcd" -->
Let's also take a look at a pressure/temperature phase space plot, for ideal gas behaviour, we'd expect

$$10^{5}P(atm) = \frac{N_{D_2}k_BT(C)}{V} + \frac{273.15N_{D_2}k_B}{V}$$

Because the number of gas molecules $N_{D_2}$ appears in both the gradient term and offset term, the motion around phase space can look quite complicated. Let's see.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="fc34463a-1e17-45db-a3e1-daee61c25494" outputId="6330a79c-da3d-403e-ff06-2a1ad051a000"
# Create a scatter plot of pressure vs. temperature
plt.figure(figsize=(6, 6))
plt.scatter(combined_df['Inferred mean T (C)'], combined_df['Pressure (Bar)'], marker=".", color='orange', alpha=0.7)

# Label axes
plt.xlabel('Inferred mean T (C)')
plt.ylabel('Pressure (bar)')

# Add title
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")

# Display the plot
plt.show()
```

<!-- #region id="bbee463c-41c5-4f81-864e-a9c479e0441e" -->
If we down-sample the data a bit, we can get a bit more sense of the dynamics from the scatter plot. It also allows us to see potentially interesting points that we'd like to pay more attention to. For example:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 805} id="5e5cf7e2-194f-4b33-a2d2-ae574249e6ca" outputId="62b29003-fd24-4918-9c1b-56e24a03fdc0"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Power (W)','Inferred mean T (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                   "Inferred mean T (C)", "Pressure (Bar)",
                        colors=['orange','blue', 'green', 'red'], downsample=100)
```

<!-- #region id="a618e075-9877-4207-b5c3-27c9bcbdec19" -->
Let's now revisit the P-T phase diagram and this time overlay what we would expect from the ideal gas law.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="7c1101f9-0ed9-423a-941f-cfdc98f3154b" outputId="80758136-697b-4397-d3f1-7e1b20352d7b"
downsample = 100

# Downsample the data
combined_df_downsampled = combined_df.iloc[::downsample]

# Create a scatter plot of pressure vs. temperature
plt.figure(figsize=(6, 6))
plt.scatter(combined_df_downsampled['Inferred mean T (C)'], combined_df_downsampled['Pressure (Bar)'], marker=".", color='orange', alpha=0.7)

# Label axes
plt.xlabel('Inferred mean T (C)')
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

temperatures_C = combined_df_downsampled['Inferred mean T (C)']
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

```python id="u1VAlAmYzBoe"

```
