---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: Python 3
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/2025-01-20-sem-eds-data-example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/2025-01-20-sem-eds-data-example.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>

```python id="mV2cfXsK9ORB"
import gdown
import h5py
import numpy as np
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
```

```python id="d8pqg2snyo25"
# Google Drive file URL
#file_id = "1Qe67DEDiPcuzZ_5MH9C-wVPrq1VHyDz8"  # Extracted file ID from your URL
#filename = "Project 1 Specimen 1 Area 2 Site 1 Map Data 11.h5oina"

file_id = "1Qtd9BD_U5W7TxXMVeBNBWXf2IqDzeoTS"  # Extracted file ID from your URL
filename = "Project 1 Specimen 1 Area 2 Site 20 Map Data 30.h5oina"
```

```python colab={"base_uri": "https://localhost:8080/", "height": 122} id="qv2gy7wMBoQB" outputId="59ce9fc3-393c-4adf-f79d-c2a65a7b0afc"
url = f"https://drive.google.com/uc?id={file_id}"  # Construct direct download link
output = filename

# Download file
gdown.download(url, output, quiet=False)
```

```python id="J6-4hmceyzV0"
file = h5py.File(output, 'r')
```

```python colab={"base_uri": "https://localhost:8080/"} id="E-ee_jFc7y1t" outputId="ed7a9682-9b06-4ea0-e6af-8e6216af304d"
print(list(file.keys()))  # List contents of the file
```

```python colab={"base_uri": "https://localhost:8080/"} id="MwLs11_Aypeb" outputId="55c741c0-3215-4dbd-9c77-b0ee3e7775a7"
spectrum_Data = file['/1/EDS/Data/Spectrum']
spectrum_Data.size
```

```python colab={"base_uri": "https://localhost:8080/"} id="EdpnwJTnyuli" outputId="e004eaeb-10a4-4ce8-dc1b-f25cd30ebc9d"
spectrum_Data.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ov1WzX0c8D1j" outputId="68f48b2c-0f32-4253-c93f-5ba08946bd37"
X_Data = file['/1/EDS/Data/X']
X_Data.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="kl95H3pC8F9b" outputId="5f96f7f2-b3ff-4356-ad8f-6618fb93e694"
Y_Data = file['/1/EDS/Data/Y']
Y_Data.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="0bHebYlQ8TkV" outputId="49609fdc-03de-4b0b-f106-4cd6521ea26b"
X_Data[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="S3A_XUiA8KEq" outputId="cd53a11f-59bf-48dc-e324-72c43ab97847"
maxx_micron = X_Data[X_Data.size-1]
maxx_micron
```

```python id="0hiofQek9X_h"
scaling_factor = 0.316162  # Microns per pixel
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ktj6Lo3r8OBG" outputId="5cc240ef-ca4a-41b6-fecd-dbcd71149cd9"
Y_Data[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="u_P115MK8XDW" outputId="a54fb23a-bc8e-418d-8025-d87f266834c3"
maxy_micron = Y_Data[Y_Data.size-1]
maxy_micron
```

```python colab={"base_uri": "https://localhost:8080/"} id="l_ozJmnI8Za-" outputId="7edfba07-234b-45c4-cb14-39cea3bca265"
spectrum_Data[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="lsIC5OhG94lu" outputId="dbf89050-ea79-4de0-c067-87638cfd6b7f"
# Determine the grid size
x_pixels = len(np.unique(file['/1/EDS/Data/X']))  # Number of unique X positions
y_pixels = len(np.unique(file['/1/EDS/Data/Y']))  # Number of unique Y positions
spectrum_length = spectrum_Data.shape[1]  # Length of each spectrum
print(f"Grid size: {y_pixels} x {x_pixels} pixels.")
print(f"Spectrum length: {spectrum_length} channels.")
```

```python colab={"base_uri": "https://localhost:8080/"} id="L6PJKyH699hq" outputId="03368639-4fea-4ab5-ab0b-277fd078fd03"
scaling_factor_x = maxx_micron/x_pixels
scaling_factor_x
```

```python colab={"base_uri": "https://localhost:8080/"} id="2TjfRBu4-FLD" outputId="6d57717a-55fc-4949-deca-01359964041b"
scaling_factor_y = maxy_micron/y_pixels
scaling_factor_y
```

```python colab={"base_uri": "https://localhost:8080/"} id="UTKEounG8bde" outputId="c59800a4-36a8-4058-cb95-29a89329b98d"
# Open the file and access datasets
print("Opening HDF5 file and preparing for chunked processing...")

spectra_array2 = np.zeros((y_pixels, x_pixels, spectrum_length), dtype=np.uint8)

# Process spectra in chunks
print("Processing spectra in chunks...")
chunk_size = 10000  # Adjust based on your system's memory
for start_idx in tqdm(range(0, spectrum_Data.shape[0], chunk_size), desc="Processing chunks"):
  end_idx = min(start_idx + chunk_size, spectrum_Data.shape[0])
  chunk = spectrum_Data[start_idx:end_idx, :]  # Load chunk into memory

  # Map 1D chunk indices to 2D grid indices
  for flat_idx, spectrum in enumerate(chunk):
    y_idx = (start_idx + flat_idx) // x_pixels
    x_idx = (start_idx + flat_idx) % x_pixels
    spectra_array2[y_idx, x_idx, :] = spectrum

print("Spectra processing complete.")
```

```python id="g9y9CpV0-To3"
np.savez_compressed("spectra_array2_compressed.npz", spectra_array2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="qXc3XTb0_dAt" outputId="eb6dd7e1-c3c3-4d35-ba21-2f614791af3c"
# Aggregate the spectrum for the bottom-right quadrant
cumulative_spectrum = np.sum(spectra_array2, axis=(0, 1))

# Plot the spectrum for the bottom-right quadrant
plt.figure(figsize=(8, 6))
plt.plot(cumulative_spectrum, label="cumulative_spectrum")

# Highlight the channels 150 to 180 with a semitransparent rectangle
plt.axvspan(140, 170, color='orange', alpha=0.3, label="Highlighted Channels")

plt.title("Aggregated Spectrum")
plt.xlabel("Channel")
plt.ylabel("Intensity")
plt.legend()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 663} id="duFQZCWd_81B" outputId="c76f449e-903d-412e-e9e3-dfcab365894f"
# Define the subset of channels (140 to 170 inclusive)
channel_start = 140
channel_end = 170

# Sum the counts across the subset of channels for every pixel
subset_sums = np.sum(spectra_array2[:, :, channel_start:channel_end + 1], axis=2)

# Normalize the sums to avoid zero values (Log scale requires positive values)
subset_sums = subset_sums + 1  # Add a small offset to avoid log(0)

# Plot the darkness map with a log scale
plt.figure(figsize=(10, 8))
plt.imshow(subset_sums, cmap="gray", norm=LogNorm(vmin=subset_sums.min(), vmax=subset_sums.max()), origin="lower")
plt.colorbar(label="Darkness Value (Log Scale)")
plt.title("Darkness Map (Log Scale for Channels 140 to 170)")
plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="jpZunCr5AKFO" outputId="a7609421-4dba-4f7f-d1eb-854b0c6956bc"
file['/1/EDS/Header/Stage Position/X'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="EdzVafVBDbv-" outputId="40321ee0-35c1-42ab-865d-1aeee6a20911"
file['/1/EDS/Header/Stage Position/Y'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="gJdM9yEwEJho" outputId="0b2aec49-37e5-42d9-856a-b63f41db6f08"
file['/1/EDS/Header/Stage Position/Z'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="5A9-7hF7DiY1" outputId="9fe4dd67-ca5d-4e2e-c77c-1548bda3699b"
file['/1/EDS/Header/X Cells'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="opsNxPMyD4RY" outputId="eb8bba05-6eff-4877-cf9f-3c2e290093ac"
file['/1/EDS/Header/Y Cells'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="lSguxSGBD2Zl" outputId="35ec567a-9d19-4883-8d80-ca9f94194cb4"
file['/1/EDS/Header/X Step'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="_LFDhI5QD8MH" outputId="abc6dbf2-e870-4d25-fb5a-aedd83898588"
file['/1/EDS/Header/Y Step'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="BvuTrRPgD_CJ" outputId="a1623b2c-ba95-4d52-f25f-0505d03929e2"
image_data = file['/1/Electron Image/Data/SE']
image_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="0wMBS4twEmif" outputId="69a74b2d-2e49-45a1-ec5b-dfee87783282"
group_path = '/1/Electron Image/Data/SE'

# List all child keys in the group
children = list(file[group_path].keys())
print(f"Children of group '{group_path}': {children}")

# Access the first child (for example) if needed
first_child = children[0]
print(f"First child in group: {first_child}")

# Access data within the first child
data = file[group_path][first_child]
print(f"Data type: {type(data)}")
print(f"Data shape (if dataset): {getattr(data, 'shape', 'Not a dataset')}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="GWI97zFeFJWA" outputId="cdd73b30-96aa-4c8a-b781-9482b714a98f"
data
```

```python id="sqaVZwH_FPxB"
group_path = '/1/Electron Image/Data/SE'
first_child = list(file[group_path].keys())[0]  # Access the first child dynamically
data = file[group_path][first_child][:]  # Load the dataset as a NumPy array
```

```python colab={"base_uri": "https://localhost:8080/"} id="rcTOR2wfFg0c" outputId="cabed3c4-47fa-4cee-cc63-9aa42bfdc0bc"
data.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="KoVYWh5fGKsF" outputId="bddfffd5-81b3-4e33-fef0-0af0dba9fecc"
file['/1/Electron Image/Header/X Cells'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="KbTPZ3VQGQfo" outputId="5a87f926-034e-4808-836d-b98e4c5eeacb"
file['/1/Electron Image/Header/Y Cells'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="10-P5-yYI9-8" outputId="b2a4dc4e-4d5a-44fe-cf07-9671ccc0427c"
file['/1/Electron Image/Header/X Step'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="GlzOFc_6I-MX" outputId="4641ea54-9b70-412f-a8c6-8ed1f4433742"
file['/1/Electron Image/Header/Y Step'][0]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 592} id="GmWZ0uT-FlI8" outputId="e0c2150c-56ea-4ab0-92fd-25a14d45a0b8"
# Reshape the data using the correct dimensions (height = 1408, width = 2048)
width, height = 1024, 704  # Dimensions from your provided data
data_reshaped = data.reshape(height, width)

# Plot the reshaped image
plt.figure(figsize=(10, 7))  # Adjust figure size as needed
plt.imshow(data_reshaped, cmap='gray')  # Use a grayscale colormap for SEM images
plt.title('SEM Image')
plt.axis('off')  # Hide axes for better visualization
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="qyMgyblCIPnG" outputId="bbb0bc9b-de82-4c53-f781-7eb554ec4377"
# Use plt's normalization for consistent appearance
fig, ax = plt.subplots()
im = ax.imshow(data_reshaped, cmap='gray')
plt.axis('off')  # Turn off the axes

# Save the image directly from the matplotlib figure
output_filename = "sem_image_match.jpg"
fig.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)  # Adjust dpi for resolution
plt.close(fig)  # Close the figure to free up memory

print(f"Image saved as {output_filename} with appearance matching plt.imshow.")
```

```python id="foXsyVRZIuqd"
np.savez_compressed("data_reshaped.npz", data_reshaped)
```

```python id="qgKituU-JfPA"

```
