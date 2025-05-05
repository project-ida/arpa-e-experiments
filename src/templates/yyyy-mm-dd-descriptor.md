---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="3f826b5b-1cf7-45b2-9622-89c10dbf1eb2" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/templates/yyyy-mm-dd-descriptor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/templates/yyyy-mm-dd-descriptor.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="YVrAMdOsLZXM" -->
## READ FIRST
1. Follow instructions in the README in the templates folder
2. Leave the Cloab and Jupyter links above alone. Github will make sure they point to where you save the file as long as they are right at the top
3. Carry on with your analysis
4. If you are using colab, don't forget to **save your changes** otherwise they'll not be commited back to the Github repo
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
# Descriptive title
<!-- #endregion -->

<!-- #region id="487e78f6-0666-4d0c-ade0-30403aa31975" -->
Description of the experiment
<!-- #endregion -->

```python id="6e5640a1-12da-4157-a5e8-5f73f882e6a7"
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
# - load_data
from libs.helpers import *

# Necessary for using load_data on password protected data urls
# - authenticate
# - get_credentials
from libs.auth import *
```

```python id="961e61cb-8a0c-4f45-9c9e-ec6c81441524"
meta = {
    "descriptor" : "CHANGE THIS" # This will go into the title of all plots
}
```

<!-- #region id="d1d7c4fc-7df2-4c54-8be1-2750a9071260" -->
## Reading the raw data
<!-- #endregion -->

```python id="fde663ef-7691-4c50-8a21-df4e77c67d25"
# Read the data from remote source that includes headers
example_df = load_data('CHANGE THIS TO DATA URL')

# Manually read the data from source that does NOT include headers
# CHANGE "names" to be descriptive of the measurements
# e.g. pressure data  names=['time', 'Voltage1', 'Voltage2', 'Voltage3', 'Voltage4']
example_df = pd.read_csv(
    '20240923_192738_Ti_etched_run_2_cycles+RTeq.csv',
    names=['time', 'CHANGE', 'CHANGE', 'CHANGE'],
    parse_dates=['time'],
    date_format="ISO8601",
    index_col='time',
    header=None
)
```

```python id="ba674648-c367-44a9-a73f-dce6c66cfdf2"
# Print out basic description of the data, including the start and end times of the data, total number of data points,
# average time between measurements, and a count of NaN values for each column.
print_info(example_df)
```

```python id="e3fd770c-4082-432d-85c6-676c0ffdb901"
plt.figure(figsize=(8, 4))
plt.plot(example_df['CHANGE THIS TO COLUMN NAME'])
plt.xlabel('Time')
plt.ylabel('CHANGE THIS TO NAME AND UNITS')
plt.xticks(rotation=45)
plt.title(f"{meta['descriptor']} {example_df.index[0].date()}")
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

We created a reusable helper function for this.
<!-- #endregion -->

```python id="e0fa56de-0ec6-4b9c-b2a7-4da02eb09812"
# Adding the meta data as the second argument gives plotting functions access to e.g. experiment descriptor for titles
combined_df = process_data([DATAFRAME_1, DATAFRAME_2, ETC], meta)
```

```python id="4dc684a8-34e1-4265-ae77-4a0eefcaa494"
combined_df.head()
```

<!-- #region id="5a570c69-908c-4530-a30d-1f0a67b5c60e" -->
**INCLUDE ANY ADDITIONAL PROCESSING HERE**
<!-- #endregion -->

```python id="a68f9cd1-e486-4c50-bb93-30feff9f1076"
# INCLUDE ANY ADDITIONAL PROCESSING HERE
```

<!-- #region id="7f7d99e7-0c45-48f4-9783-d82773ebd25f" -->
## Visualising the data

We created some reusable plotting functions, `plot_panels` and `plot_panels_with_scatter`. Full documentation is avaiable through `shift+tab` over the function name once it's imported. See also the doc string in `libs\helper.py`  

For `plot_panels`, it allows you to choose:
- What columns to display (required)
- Start time (optional)
- Stop time (optional)
- Colors (optional)
- Path to save the figure (optional)
- Downsampling (optional)
- A maker to highlight a specific time with a vertical line
<!-- #endregion -->

```python id="4022b73b-c6d1-4bf8-b8f1-9529f35f23dc"
# Example usage
fig, axes = plot_panels(combined_df, ['column_1', 'column_2', 'column_3','ETC'],
                      start="2024-09-23 19:37:42", stop="2024-09-24 13:37:42",
                      save_path="plot.png", colors=['blue', 'green', 'red'],
                        downsample=10, marker="2024-09-23 23:00")

axes[0].set_ylabel("Custom Label") # Modify label after plotting, if needed
```

<!-- #region id="c5c3a60b-0625-47ef-a3a9-89fd09fc45fb" -->
For `plot_panels_with_scatter`:
- It makes side by side panels plots with a scatter plot
- You can save the static figure with the `save_path` variable just as with `plot_panels`.
- You can also add a `marker` to this function as with `plot_panels` which will add a blue dot on the scatter plot corresponding to the vertical lines on the panel plots.
- You can animate it over time.
- Note that `frames` and `animate` go together. If `animate=True` then frames determine the interval for the animation, setting how many points to skip between frames.
- The animation is saved in the `media` folder with a file name based on the date of the first measurement and the descriptor in the meta data.
<!-- #endregion -->

```python id="d5369e89-0bc0-41d9-9c60-c9674e2aa141"
# Example usage
fig, time_axes, ax_scatter = plot_panels_with_scatter(
    combined_df, ['column_1', 'column_2', 'column_3', 'ETC'], scatter_x='column_3', scatter_y='column_1',
    start="2024-09-23 19:37:42", stop="2024-09-24 13:37:42", colors=['blue', 'green'],
    downsample=10, animate=True, frames=10
```

```python id="4de79691-69b0-4af4-a9ff-b67674c1284b"
# If working in colab, then set embed=True
Video("media/file name of animation.mp4", embed=False, width=800)
```
