# Function to load metadata from a .txt file into a dictionary
def load_metadata(txt_filepath):
    metadata = {}
    with open(txt_filepath, 'r') as file:
        for line in file:
            # Split the line into key and value at the first colon
            key, value = line.strip().split(':', 1)
            key = key.strip()
            value = value.strip()

            # Attempt to convert value to an integer or float if applicable
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string if it cannot be converted

            metadata[key] = value
    return metadata