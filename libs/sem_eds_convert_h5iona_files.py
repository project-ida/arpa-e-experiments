import os
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define directories
input_folder = "h5oina"
output_folder = "pythondata"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to process a single .h5oina file
def process_h5oina_file(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0]

    print(f"Processing file: {filepath}")

    # Open the .h5oina file
    with h5py.File(filepath, 'r') as file:
        print("Opened H5 file successfully.")

        # Extract SEM image data
        print("Extracting SEM image data...")
        group_path = '/1/Electron Image/Data/SE'
        first_child = list(file[group_path].keys())[0]  # Access the first child dynamically
        imagedata = file[group_path][first_child][:]  # Load the dataset as a NumPy array

        # Reshape the data using correct dimensions
        print("Reshaping SEM image data...")
        imagewidth = file['/1/Electron Image/Header/X Cells'][0]
        imageheight = file['/1/Electron Image/Header/Y Cells'][0]
        data_reshaped = imagedata.reshape(imageheight, imagewidth)

        # Plot and save the reshaped image using matplotlib with exact resolution
        print(f"Saving SEM image as PNG: {filename}_sem.png")
        fig, ax = plt.subplots(figsize=(imagewidth / 100, imageheight / 100))  # Match original pixel dimensions
        im = ax.imshow(data_reshaped, cmap='gray')
        plt.axis('off')  # Turn off the axes
        sem_image_path = os.path.join(output_folder, f"{filename}_sem.png")  # Save as PNG for lossless compression
        fig.savefig(sem_image_path, bbox_inches='tight', pad_inches=0, dpi=100)  # Ensure 1-to-1 pixel mapping
        plt.close(fig)  # Close the figure to free up memory

        # Save SEM data as .npz
        print(f"Saving SEM data as NPZ: {filename}_sem.npz")
        sem_npz_path = os.path.join(output_folder, f"{filename}_sem.npz")
        np.savez_compressed(sem_npz_path, sem_data=data_reshaped)

        # Extract EDS spectrum data and process into a 3D array
        print("Extracting and processing EDS spectrum data...")
        spectrum_data = file['/1/EDS/Data/Spectrum'][:]
        x_pixels = file['/1/EDS/Header/X Cells'][0]
        y_pixels = file['/1/EDS/Header/Y Cells'][0]
        spectrum_length = file['/1/EDS/Header/Number Channels'][0]

        spectra_array2 = np.zeros((y_pixels, x_pixels, spectrum_length), dtype=np.uint8)

        chunk_size = 10000  # Adjust based on your system's memory
        for start_idx in tqdm(range(0, spectrum_data.shape[0], chunk_size), desc="Processing chunks"):
            end_idx = min(start_idx + chunk_size, spectrum_data.shape[0])
            chunk = spectrum_data[start_idx:end_idx, :]  # Load chunk into memory

            # Map 1D chunk indices to 2D grid indices
            for flat_idx, spectrum in enumerate(chunk):
                y_idx = (start_idx + flat_idx) // x_pixels
                x_idx = (start_idx + flat_idx) % x_pixels
                spectra_array2[y_idx, x_idx, :] = spectrum

        eds_npz_path = os.path.join(output_folder, f"{filename}_eds.npz")
        print(f"Saving EDS data as NPZ: {filename}_eds.npz")
        np.savez_compressed(eds_npz_path, eds_data=spectra_array2)

        # Extract metadata
        print("Extracting metadata...")
        metadata = {
            '/1/Electron Image/Header/X Cells': imagewidth,
            '/1/Electron Image/Header/Y Cells': imageheight,
            '/1/Electron Image/Header/X Step': file['/1/Electron Image/Header/X Step'][0],
            '/1/Electron Image/Header/Y Step': file['/1/Electron Image/Header/Y Step'][0],
            '/1/EDS/Header/X Cells': x_pixels,
            '/1/EDS/Header/Y Cells': y_pixels,
            '/1/EDS/Header/X Step': file['/1/EDS/Header/X Step'][0],
            '/1/EDS/Header/Y Step': file['/1/EDS/Header/Y Step'][0],
            '/1/EDS/Header/Start Channel': file['/1/EDS/Header/Start Channel'][0],
            '/1/EDS/Header/Channel Width': file['/1/EDS/Header/Channel Width'][0],
            '/1/EDS/Header/Energy Range': file['/1/EDS/Header/Energy Range'][0],
            '/1/EDS/Header/Number Channels': spectrum_length,
            '/1/EDS/Header/Stage Position/X': file['/1/EDS/Header/Stage Position/X'][0],
            '/1/EDS/Header/Stage Position/Y': file['/1/EDS/Header/Stage Position/Y'][0],
            '/1/EDS/Header/Stage Position/Z': file['/1/EDS/Header/Stage Position/Z'][0]
        }
        metadata_path = os.path.join(output_folder, f"{filename}_metadata.txt")
        print(f"Saving metadata as TXT: {filename}_metadata.txt")
        with open(metadata_path, 'w') as meta_file:
            for key, value in metadata.items():
                meta_file.write(f"{key}: {value}\n")

    print(f"Finished processing file: {filename}\n")

# Iterate through files in the input folder, skipping the first 45
file_count = 0
files_processed = 0
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".h5oina") and "Montaged" not in file:
            file_count += 1
            if file_count <= 45:
                continue  # Skip the first 45 files
            filepath = os.path.join(root, file)
            print(f"Starting processing for file: {file}")
            process_h5oina_file(filepath)
            files_processed += 1
            #if files_processed >= 2:
            #    print("Processed 2 files. Stopping.")
            #    break
    #if files_processed >= 2:
    #    break

print("All specified files have been processed.")
