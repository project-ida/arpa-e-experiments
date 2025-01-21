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

<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/tutorials/working-with-sem-eds-data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/tutorials/working-with-sem-eds-data.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# Working with scanning electron microscope data


We are working with people at Texas Tech to characterise any morphological and elemental changes that take place as part of the our experiments. This involves using a scanning electron microscope to generate pictures of the material surface (so called SEM data) and also energy-dispersive X-ray spectroscopy (EDS) to analyze elemental composition.

The data is exported from [Oxford Instruments Aztec software](https://nano.oxinst.com/products/aztec/) into a HDF5 like format called [h5iona](https://github.com/oinanoanalysis/h5oina) that's stored in Google Drive, (e.g. [this folder](https://drive.google.com/drive/folders/1WqbhoVJ5d6HWgWa1Bo3wYWzeZD6t-ia5)). This compressed data format is a pain to work with in python, so we've converted it into numpy arrays which are also stored in Google Drive (see e.g. [this folder](https://drive.google.com/drive/folders/1Zp6a3h2Es3q3eercQlK0gMA57-sqfoSo)).

The data conversion is performed using `sem-eds-convert-h5iona-files.py` which can be found in the [libs](../libs/) folder of this repository.

```python
# RUN THIS IF YOU ARE USING GOOGLE COLAB
# It pulls the arpa-e repo into colab and makes sure we can import heplers from it
# It also changes the working directory to the arpa-e folder so we can load media easier

import sys
import os
!git clone https://github.com/project-ida/arpa-e-experiments.git
sys.path.insert(0,'/content/arpa-e-experiments')
os.chdir('/content/arpa-e-experiments')
```

```python
# RUN THIS IF YOU ARE LOCAL. 
# It makes sure we can import helpers from libs which is one level up

import sys
import os

# Get the parent directory (one level up from the current directory)
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, project_root)
```

```python
import gdown
import numpy as np
import matplotlib.pyplot as plt
from libs.sem_eds_helpers import *
```

## Loading the metadata for a specific site


Inside [20241023 jonah 4 / Project 1 / pythondata](https://drive.google.com/drive/folders/1WqbhoVJ5d6HWgWa1Bo3wYWzeZD6t-ia5), we can  grab the share link for the metadata associated with site 11 - this is called `Project 1 Specimen 1 Area 1 Site 11 Map Data 10_metadata.txt`

We're using `gdown` to download the data. We wrote a separate tutorial about it [here](working-with-google-drive.ipynb).

```python
metadata_path = gdown.download("https://drive.google.com/file/d/1rmpqlh43s9DXWxK6xJ0LNWYZxcTkl-ai/view?usp=drive_link", fuzzy=True)
```

You should now be able to find a file with the name given below:

```python
print(metadata_path)
```

Let's now use one of our helpers to read in the metadata associated with the SEM and EDS data.

```python
metadata = load_metadata(metadata_path)
```

```python
# Print the loaded metadata
for key, value in metadata.items():
    print(f"{key}: {value}")
```

## Visualising a SEM image for a specific site


Now, let's grab the share link for the SEM data associated with site 11.

```python
sem_npz_path = gdown.download("https://drive.google.com/file/d/1rfAbFPRk9zwM1QNOacsbXBfGL40pjdb6/view?usp=drive_link", fuzzy=True)
```

You should now be able to find a file with the name given below:

```python
print(sem_npz_path)
```

Because the data is in numpy format, we can use `np.load` to bring the data into python.

```python
# Load the .npz file
data = np.load(sem_npz_path)
data
```

The data is stored as a kind of dictionary. We can pull out the raw data like this:

```python
sem_image_array = data['sem_data']
```

As you can see below, it's just a normal numpy array.

```python
sem_image_array
```

Let's visualise the data:

```python
# Plot the SEM image
plt.figure(figsize=(10, 7))  # Adjust the figure size as needed
plt.imshow(sem_image_array, cmap='gray')  # Use a grayscale colormap
plt.title("SEM Image")
plt.axis('off')  # Hide the axes for better visualization
plt.show()
```

## Visualising an EDS image for a specific site


We're going to follow similar steps as with with ESM data.

Let's grab the share link for the EDS data associated with site 11.

```python
eds_npz_path = gdown.download("https://drive.google.com/file/d/1rigACe3yqlH4Y1g_UWkgUFTD8ZJ2ajaH/view?usp=drive_link", fuzzy=True)
```

You should now be able to find a file with the name given below:

```python
print(eds_npz_path)
```

Load the data as before:

```python
# Load the .npz file
data = np.load(eds_npz_path)
eds_data = data['eds_data']  # Load the 3D spectrum array
```

And now we're ready to do some analysis. Let's look at the cumulative spectrum.

```python
# Calculate the cumulative spectrum by summing across all pixels
cumulative_spectrum = np.sum(eds_data, axis=(0, 1))
```

```python
# Plot the cumulative spectrum
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(cumulative_spectrum, label="Cumulative Spectrum")
plt.xlabel("Channel Index")
plt.ylabel("Intensity")
plt.title("Cumulative EDS Spectrum Across All Pixels")
plt.legend()
plt.grid(True)
plt.show()
```
