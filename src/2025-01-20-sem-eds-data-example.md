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

<!-- #region id="5485eb77-dd35-4417-9df7-c4de17127845" -->
<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/2025-01-20-sem-eds-data-example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/2025-01-20-sem-eds-data-example.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

```python id="mV2cfXsK9ORB"
import gdown
import h5py
import numpy as np
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from IPython.display import Image, display, Video, HTML
```

```python id="Heb1eebmS4eI" outputId="c5896322-26c8-45ef-f593-38ff89ec7b47" colab={"base_uri": "https://localhost:8080/"}
import sys
import os
!git clone https://github.com/project-ida/arpa-e-experiments.git
sys.path.insert(0,'/content/arpa-e-experiments')
os.chdir('/content/arpa-e-experiments')
```

```python id="d8pqg2snyo25"
# Google Drive file URL
#file_id = "1Qe67DEDiPcuzZ_5MH9C-wVPrq1VHyDz8"  # Extracted file ID from your URL
#filename = "Project 1 Specimen 1 Area 2 Site 1 Map Data 11.h5oina"

file_id = "1Qtd9BD_U5W7TxXMVeBNBWXf2IqDzeoTS"  # Extracted file ID from your URL
filename = "Project 1 Specimen 1 Area 2 Site 20 Map Data 30"
```

```python colab={"base_uri": "https://localhost:8080/", "height": 122} id="qv2gy7wMBoQB" outputId="30387f59-95e5-4013-d4f5-ce211a647010"
url = f"https://drive.google.com/uc?id={file_id}"  # Construct direct download link
output = filename+".h5oina"

# Download file
gdown.download(url, output, quiet=False)
```

```python id="J6-4hmceyzV0"
file = h5py.File(output, 'r')
```

```python colab={"base_uri": "https://localhost:8080/"} id="E-ee_jFc7y1t" outputId="3e20c7ae-9597-42a4-b57c-ff920b088a05"
print(list(file.keys()))  # List contents of the file
```

<!-- #region id="L0pKckRtT_Aw" -->
# SEM DATA
<!-- #endregion -->

```python id="sqaVZwH_FPxB"
group_path = '/1/Electron Image/Data/SE'
first_child = list(file[group_path].keys())[0]  # Access the first child dynamically
imagedata = file[group_path][first_child][:]  # Load the dataset as a NumPy array
```

```python colab={"base_uri": "https://localhost:8080/"} id="rcTOR2wfFg0c" outputId="53956406-f4e1-440c-aed9-2821cb938e1f"
imagedata.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="KoVYWh5fGKsF" outputId="7b1f402b-28b4-412c-f5d0-5e88bf8e753a"
imagewidth = file['/1/Electron Image/Header/X Cells'][0]
imagewidth
```

```python colab={"base_uri": "https://localhost:8080/"} id="KbTPZ3VQGQfo" outputId="468d941f-744a-40d9-d484-f80ac273134e"
imageheight = file['/1/Electron Image/Header/Y Cells'][0]
imageheight
```

```python colab={"base_uri": "https://localhost:8080/"} id="10-P5-yYI9-8" outputId="ce875d1e-1927-4be5-89cd-d5f6e0a51e18"
pixelsizex = file['/1/Electron Image/Header/X Step'][0]
pixelsizex
```

```python colab={"base_uri": "https://localhost:8080/"} id="GlzOFc_6I-MX" outputId="69de64e7-a91d-4c6e-c09b-0acad3a35c5a"
pixelsizey = file['/1/Electron Image/Header/Y Step'][0]
pixelsizey
```

```python colab={"base_uri": "https://localhost:8080/", "height": 592} id="GmWZ0uT-FlI8" outputId="204816e1-a5a4-4fc4-968c-aab40d87a84b"
# Reshape the data using the correct dimensions (height = 1408, width = 2048)
width, height = imagewidth, imageheight  # Dimensions from your provided data
data_reshaped = imagedata.reshape(height, width)

# Plot the reshaped image
plt.figure(figsize=(10, 7))  # Adjust figure size as needed
plt.imshow(data_reshaped, cmap='gray')  # Use a grayscale colormap for SEM images
plt.title('SEM Image')
plt.axis('off')  # Hide axes for better visualization
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="qyMgyblCIPnG" outputId="3b208a14-5369-4247-9c70-5be7b79ae19d"
# Use plt's normalization for consistent appearance
fig, ax = plt.subplots()
im = ax.imshow(data_reshaped, cmap='gray')
plt.axis('off')  # Turn off the axes

# Save the image directly from the matplotlib figure
output_filename = filename+"_sem.jpg"
fig.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)  # Adjust dpi for resolution
plt.close(fig)  # Close the figure to free up memory

print(f"Image saved as {output_filename} with appearance matching plt.imshow.")
```

```python id="foXsyVRZIuqd"
np.savez_compressed(filename+"_sem.npz", data_reshaped)
```

<!-- #region id="0iiROupgUXbe" -->
# EDS Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MwLs11_Aypeb" outputId="31edca27-ef5b-4e58-bc3e-3d7e6841d9e0"
spectrum_Data = file['/1/EDS/Data/Spectrum']
spectrum_Data.size
```

```python colab={"base_uri": "https://localhost:8080/"} id="EdpnwJTnyuli" outputId="2ed7cd06-633e-48dd-c9d9-0cfa2e45a8cc"
spectrum_Data.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ov1WzX0c8D1j" outputId="0e06ec6b-650b-4423-dc9d-84c97a8d61d4"
X_Data = file['/1/EDS/Data/X']
X_Data.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="kl95H3pC8F9b" outputId="7222b9ac-f25e-4d73-bac1-a5e673ef23d9"
Y_Data = file['/1/EDS/Data/Y']
Y_Data.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="0bHebYlQ8TkV" outputId="bb384d59-7850-4bec-9f40-055f01a2f435"
X_Data[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="S3A_XUiA8KEq" outputId="d7af32c9-b5d9-4cb1-c1db-fc4cb86c43a5"
maxx_micron = X_Data[X_Data.size-1]
maxx_micron
```

```python id="0hiofQek9X_h"
scaling_factor = 0.316162  # Microns per pixel
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ktj6Lo3r8OBG" outputId="553a4051-4524-4fab-f05a-3abf7edb3ac1"
Y_Data[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="u_P115MK8XDW" outputId="e56264f9-aff9-4b4e-db0d-f46f3b9c3403"
maxy_micron = Y_Data[Y_Data.size-1]
maxy_micron
```

```python colab={"base_uri": "https://localhost:8080/"} id="l_ozJmnI8Za-" outputId="f25c8ed3-dc27-42a1-cc5f-28eb9e4b1a6a"
spectrum_Data[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="5A9-7hF7DiY1" outputId="9a14d83a-3d91-4285-c047-0ad23ac65f8b"
file['/1/EDS/Header/X Cells'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="opsNxPMyD4RY" outputId="d069e9a1-aab3-4a6e-ee8a-a25a901e372a"
file['/1/EDS/Header/Y Cells'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="lSguxSGBD2Zl" outputId="ef97840c-c91f-4584-df14-9c56e163cb51"
file['/1/EDS/Header/X Step'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="_LFDhI5QD8MH" outputId="15a45837-0fdd-4b90-92c5-046f5ffd7ecd"
file['/1/EDS/Header/Y Step'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="lsIC5OhG94lu" outputId="3add739f-dbfd-47a1-b0c3-5ed39636345a"
# Determine the grid size
x_pixels = len(np.unique(file['/1/EDS/Data/X']))  # Number of unique X positions
y_pixels = len(np.unique(file['/1/EDS/Data/Y']))  # Number of unique Y positions
spectrum_length = spectrum_Data.shape[1]  # Length of each spectrum
print(f"Grid size: {y_pixels} x {x_pixels} pixels.")
print(f"Spectrum length: {spectrum_length} channels.")
```

```python colab={"base_uri": "https://localhost:8080/"} id="L6PJKyH699hq" outputId="21ec58b7-091e-4703-82a6-8c6fb631468c"
scaling_factor_x = maxx_micron/x_pixels
scaling_factor_x
```

```python colab={"base_uri": "https://localhost:8080/"} id="2TjfRBu4-FLD" outputId="515b0d63-e2f0-447f-d7d2-516f81c061fa"
scaling_factor_y = maxy_micron/y_pixels
scaling_factor_y
```

```python colab={"base_uri": "https://localhost:8080/"} id="UTKEounG8bde" outputId="4afed5f4-7626-447b-bd10-65010303a88b"
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
np.savez_compressed(filename+"_eds.npz", spectra_array2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="qXc3XTb0_dAt" outputId="831dc36b-f2a1-4761-bc2b-90644a1f578b"
# Aggregate the spectrum for the bottom-right quadrant
cumulative_spectrum = np.sum(spectra_array2, axis=(0, 1))

# Plot the spectrum for the bottom-right quadrant
plt.figure(figsize=(8, 6))
plt.plot(cumulative_spectrum, label="cumulative_spectrum")

plt.title("Aggregated Spectrum")
plt.xlabel("Channel")
plt.ylabel("Intensity")
plt.legend()
plt.show()
```

```python id="Vtk0EuaCwP9u" outputId="089606b9-fa8c-4b5f-8d5f-6732b948a599" colab={"base_uri": "https://localhost:8080/"}
start_channel = file['/1/EDS/Header/Start Channel'][0]
start_channel
```

```python id="0mKeobmNv9FL" outputId="edf9f8a4-b866-4ca2-cb33-bc72146e830c" colab={"base_uri": "https://localhost:8080/"}
channel_width = file['/1/EDS/Header/Channel Width'][0] # in eV
channel_width
```

```python id="R2N3inMCwAit" outputId="a5e288d2-063e-40f8-f955-35c443398a8b" colab={"base_uri": "https://localhost:8080/"}
file['/1/EDS/Header/Energy Range'][0] # in keV
```

```python id="eRK9ECTEwJFC" outputId="08fe80ea-a7b2-4b4c-adce-e65c52043cd9" colab={"base_uri": "https://localhost:8080/"}
num_channels = file['/1/EDS/Header/Number Channels'][0]
num_channels
```

```python id="dWQggoIv-O0D"
pd_start_channel = 140
pd_end_channel = 170
```

```python id="t8q5j9emyCoK" outputId="ac079ef3-fe9a-49b3-92af-9c2dbcc74022" colab={"base_uri": "https://localhost:8080/", "height": 604}
# Calculate the energy axis in keV
channels = np.arange(num_channels)
energy_keV = (start_channel + channels * channel_width) / 1000  # Convert eV to keV

# Aggregate the spectrum for the bottom-right quadrant
cumulative_spectrum = np.sum(spectra_array2, axis=(0, 1))

# Plot the spectrum with dual x-axes
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot the spectrum on the first axis
ax1.plot(channels, cumulative_spectrum, label="cumulative_spectrum", color="blue")
ax1.set_xlabel("Channel")
ax1.set_ylabel("Intensity")
ax1.legend(loc="upper right")
ax1.set_title("Aggregated Spectrum")

# Highlight the channels with a semitransparent rectangle
ax1.axvspan(pd_start_channel, pd_end_channel, color="orange", alpha=0.3, label="Highlighted Channels")
ax1.legend(loc="upper right")

# Set x-axis limits explicitly to start from the leftmost edge
ax1.set_xlim(0, num_channels - 1)  # Channels range from 0 to 1023

# Create a second x-axis for energy in keV
ax2 = ax1.twiny()

# Define energy tick marks (0 to 20 keV in increments of 1 keV)
energy_ticks_keV = np.arange(0, 21, 1)  # Energy values in keV
channel_positions = [(e * 1000 - start_channel) / channel_width for e in energy_ticks_keV]

# Set the energy ticks and labels
ax2.set_xlim(ax1.get_xlim())  # Ensure upper and lower axes are aligned
ax2.set_xticks(channel_positions)
ax2.set_xticklabels([f"{e:.0f}" for e in energy_ticks_keV])
ax2.set_xlabel("Energy (keV)")

plt.show()
```

```python id="gEGx5KW7yyC1"
def channel_to_keV(channel, start_channel, channel_width):
    """
    Convert a channel number into energy (keV).

    Parameters:
        channel (int or float): The channel number to convert.
        start_channel (float): The start channel value in eV.
        channel_width (float): The channel width in eV.

    Returns:
        float: Energy in keV.
    """
    energy_keV = (start_channel + channel * channel_width) / 1000  # Convert eV to keV
    return energy_keV
```

```python id="9fZ5eZv57Jd7" outputId="f9dcace6-b8a3-46ff-f2fe-21eb81151f19" colab={"base_uri": "https://localhost:8080/"}
peak_channel = np.argmax(cumulative_spectrum)
peak_channel
```

```python id="UH5bdGGF7NVU" outputId="d4b199a0-9c9b-4685-9ad9-1e6416d5c9cc" colab={"base_uri": "https://localhost:8080/"}
energy = channel_to_keV(peak_channel, start_channel, channel_width)
print(f"Channel {peak_channel} corresponds to {energy:.2f} keV.")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 663} id="duFQZCWd_81B" outputId="883abc4a-7232-4524-d544-0ee6d69c549c"
# Sum the counts across the subset of channels for every pixel
subset_sums = np.sum(spectra_array2[:, :, pd_start_channel:pd_end_channel + 1], axis=2)

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

<!-- #region id="fQpyH_R_UkoJ" -->
# STICHING DATA
<!-- #endregion -->

```python id="kF5XBB9_Unmz" outputId="b0b20085-2a67-485e-a918-2cc789231287" colab={"base_uri": "https://localhost:8080/"}
file['/1/EDS/Header/Stage Position/X'][0] # in mm
```

```python id="Tg1RNEV6Uonk" outputId="373eecb2-3029-4f87-fbb1-856dc4e6ad08" colab={"base_uri": "https://localhost:8080/"}
file['/1/EDS/Header/Stage Position/Y'][0] # in mm
```

```python id="WRvByBeEUp2F" outputId="5381e4bb-7e2a-4c18-c611-12a0c4ab63bc" colab={"base_uri": "https://localhost:8080/"}
file['/1/EDS/Header/Stage Position/Z'][0] # in mm
```

```python id="rtVf4wv6U5ct" outputId="606b3472-d0b4-4c9c-d4c6-2094ea8567fb" colab={"base_uri": "https://localhost:8080/", "height": 777}
Image(filename="media/aztec-01.png", width=1474, height=760)
```

```python id="HqPQcgVaU6hd" outputId="bf682df6-a11e-47ad-c83e-918ab8e87150" colab={"base_uri": "https://localhost:8080/", "height": 777}
Image(filename="media/aztec-02.png", width=1474, height=760)
```

```python id="ZMVDjKdHVCc1" outputId="353c4089-24ff-4024-f9d3-9fe7f8cdf580" colab={"base_uri": "https://localhost:8080/", "height": 777}
Image(filename="media/aztec-03.png", width=1474, height=760)
```

```python id="L-1smG3nVCrS" outputId="384be0df-7990-405a-dd7a-9a7494653779" colab={"base_uri": "https://localhost:8080/", "height": 777}
Image(filename="media/aztec-04.png", width=1474, height=760)
```

```python id="c4jawa8FVC6r" outputId="bf50b28d-3789-441c-cf70-637fb677070e" colab={"base_uri": "https://localhost:8080/", "height": 777}
Image(filename="media/aztec-05.png", width=1474, height=760)
```

```python id="ZFGDOIuEWinx"

```
