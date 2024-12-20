---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="3f826b5b-1cf7-45b2-9622-89c10dbf1eb2" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/2024-12-13-palladium-foil.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/2024-12-13-palladium-foil.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
# 2024-12-13 Palladium foil
<!-- #endregion -->

<!-- #region id="487e78f6-0666-4d0c-ade0-30403aa31975" -->
A 103.8 mg Palladium foil is gas loaded with deuterium, in a 0.59L chamber.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6e5640a1-12da-4157-a5e8-5f73f882e6a7" outputId="262a5151-f8e9-4686-add5-932e85b45993"
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
    'http://nucleonics.mit.edu/csv-files/new-chamber-first-pd-5.csv',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="686ac467-0ef2-48a8-9598-a80f357130a8" outputId="e57991f2-64b8-431b-8cb6-2de4d277dcd9"
# Print out basic description of the data, including any NaNs
print_info(temperature_df)
```

<!-- #region id="6e7a2688-aa5a-4533-a75e-f86815179b75" -->
We're interested in using `chamber-RTD1` and `chamber-RTD2` to get a sense of the tempreature near the sample. Let's see how close they are to one another.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="xW-mQwpYrzIn" outputId="a4148350-a885-4ec8-e501-8ade65179e88"
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

<!-- #region id="75744b06-6ec3-49cc-a1db-8d7b075b176f" -->
### Heating power
<!-- #endregion -->

```python id="3a6692b1-462c-43de-83de-d31864a847b3"
# Read the heating power data
heating_df = pd.read_csv(
    'http://nucleonics.mit.edu/csv-files/new-chamber-first-pd-1.csv',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time'
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="81174345-4f9b-4095-ab2e-3dfd172ea8a2" outputId="1e921d27-f6cf-47ca-fe8c-4c2602f69082"
# Print out basic description of the data, including any NaNs
print_info(heating_df)
```

<!-- #region id="zlFIC6oEFybb" -->
We'll rename "Output power" to just "Power" convenience in plotting
<!-- #endregion -->

```python id="_tu83TMFFpu6"
heating_df.rename(columns={'Output power (W)': 'Power (W)'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="f72ac15c-1a1f-4855-b3ea-eff6979553ce" outputId="03821771-6651-47a0-a078-99ecfe02c7bf"
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

```python id="D8kU4YY7b3W7"
# Read the pressure data
pressure_df = pd.read_csv(
    'http://nucleonics.mit.edu/csv-files/new-chamber-first-pd-3.csv',
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time',
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="5a016413-224d-4ffd-bdd5-1e403830f278" outputId="61b7617c-3ccc-495e-9019-80ec82310415"
# Print out basic description of the data, including any NaNs
print_info(pressure_df)
```

<!-- #region id="rbdWZx4qxkUl" -->
We'll rename the pressure and remove the Ch3 for convenience in plotting.
<!-- #endregion -->

```python id="4dhrl8p8xjW9"
pressure_df.rename(columns={'Pressure Ch3 (bar)': 'Pressure (Bar)'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="hmG1mduzeawm" outputId="833ea25f-3fca-4dcd-c720-97dc4c4c7ab0"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="xip8p2F2LYMF" outputId="1c2615d7-31e1-422c-99e0-5da53feea3a5"
pressure_df['Pressure (Bar)']['2024-12-13 18:30':'2024-12-13 18:34'].plot(ylabel="Pressure (Bar)");
```

<!-- #region id="jIFug-smLgEy" -->
When we combine the data, we'll cut off this earlier time and right at the end so that we can more easily calculate the deuterium loading.
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

```python colab={"base_uri": "https://localhost:8080/", "height": 341} id="ASNKpZWVlcFR" outputId="7f35b6f1-e0bb-4d74-8312-37d19b03b55b"
combined_df.head()
```

```python id="OMJWLq_rLy9I"
# Cutting off the earlier times before the experiment was properly started
combined_df = combined_df['2024-12-13 18:32':'2024-12-20 00:00']
```

<!-- #region id="ea652753-5d8e-4a87-9d2c-868577ff1b70" -->
### Inferring deuterium loading

The deuterium loading is inferred by using the ideal gas law to calculate how many deuterium molecules are present in the gas over time:

$$N_{D_2} = \frac{PV}{k_B T}$$

and associating any changes $\Delta N_{D_2}$ with deuteium entering the lattice. We can then calculate the loading based on the number of lattice atoms $N_{lattice}$:

$$N_{lattice} = \frac{m_{sample}}{m_{Pd}}$$

$$N_{lattice} = \frac{103.8\times 10^{-6}}{1.77\times 10^{-25}}$$

The loading is then:

$$2\frac{\Delta N_{D_2}}{N_{lattice}}$$

The factor 2 arrises because a single $D_2$ molecule becomes 2 deuterons once inside the lattice.
<!-- #endregion -->

```python id="3c4e4339-bd7f-4632-9c8d-21aa23729b88"
# Constants
V = 0.59 / 1000  # Volume of the container in m^3
kB = 1.3806503e-23  # Boltzmann constant in J/K
N_lattice = 103.8e-6 / 1.77e-25
```

```python id="64f11bfe-a2d9-45b9-8069-3206860299e2"
combined_df['$D_2$ molecules'] = (combined_df['Pressure (Bar)']*1e5 * V) / (kB * (combined_df['Temperature (C)'] + 273.15))
```

```python id="56c3bcb5-de7a-4a91-8583-3922d6f47f00"
combined_df['D/Pd Loading'] = 2*(combined_df.iloc[0]['$D_2$ molecules'] - combined_df['$D_2$ molecules']) / N_lattice
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="6d44a2b5-ddd6-4024-b281-dbd293f6c85c" outputId="18fc0784-8d6e-4fbd-eddd-e42e15587aa6"
plt.figure(figsize=(8, 4))
plt.plot(combined_df['$D_2$ molecules'])
plt.xlabel('Time')
plt.ylabel('$D_2$ molecules')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {combined_df.index[0].date()}")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 463} id="6abfec40-461f-46e7-a987-00218be5b56f" outputId="9ac8f040-6ce4-4279-94f5-f0ceb576e0de"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 806} id="4022b73b-c6d1-4bf8-b8f1-9529f35f23dc" outputId="8d63c8ee-36d0-4884-a210-37eeffb827de"
fig, axes = plot_panels(combined_df, ['Power (W)','Temperature (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                        colors=['orange','blue', 'green', 'red'])
```

<!-- #region id="e26bfe5b-772c-48f8-a363-5a770aa5afcd" -->
Let's also take a look at a pressure/temperature phase space plot, for ideal gas behaviour, we'd expect

$$10^{5}P(atm) = \frac{N_{D_2}k_BT(C)}{V} + \frac{273.15N_{D_2}k_B}{V}$$

Because the number of gas molecules $N_{D_2}$ appears in both the gradient term and offset term, the motion around phase space can look quite complicated. Let's see.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="fc34463a-1e17-45db-a3e1-daee61c25494" outputId="dc2b34ab-5c48-4bdf-be91-d871e57f9cfc"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 805} id="5e5cf7e2-194f-4b33-a2d2-ae574249e6ca" outputId="4343dd4a-8bc6-41ec-e716-cefc79837e12"
fig, panel_axes, scatter_axes = plot_panels_with_scatter(combined_df, ['Power (W)','Temperature (C)', 'Pressure (Bar)', 'D/Pd Loading'],
                   "Temperature (C)", "Pressure (Bar)",
                        colors=['orange','blue', 'green', 'red'], downsample=100)
```

<!-- #region id="a618e075-9877-4207-b5c3-27c9bcbdec19" -->
Let's now revisit the P-T phase diagram and this time overlay what we would expect from the ideal gas law.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="7c1101f9-0ed9-423a-941f-cfdc98f3154b" outputId="8d5142b8-81c8-43e7-f735-192b2a6e4cb7"
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
