import rioxarray
import os
import pandas as pd
import numpy as np
import glob
import time
import dask
from dask.delayed import delayed

# Record the start time
start_time = time.time()

# Output path
out = '/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/parquet_files'
os.makedirs(out, exist_ok=True)

# Code to read in all training data
all_files = glob.glob('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj/*.tif')

# Function to process each file
@delayed
def process_file(f):
    try:
        # Read in file and convert to numpy
        in_file = rioxarray.open_rasterio(f).to_numpy().astype(float).round(3)

        # Convert to band last
        in_file = np.moveaxis(in_file, 0, 2)

        x = in_file[:, :, :-1]
        x[x == 0] = np.nan
        x = np.round(x, 2)

        y = in_file[:, :, -1].astype(float)
        y[y < 0] = 0
        y[y > 1] = 0

        stacked = np.dstack([x, y])

        # Reshape the 3D matrix to 2D
        rows, cols, bands = stacked.shape
        reshaped_data = stacked.reshape(rows * cols, bands)

        band_names = ['blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2', 'dNBR', 'dNDVI', 'dNDII', 'y']

        # Create a DataFrame
        training = pd.DataFrame(reshaped_data, columns=band_names).dropna()

        return training

    except Exception as e:
        return pd.DataFrame()  # Return an empty DataFrame on error

# Process and save each file incrementally to avoid memory errors
for i, f in enumerate(all_files):
    delayed_training = process_file(f)
    df = delayed_training.compute()  # Compute the delayed result (process the file)
    
    # Save each DataFrame to a separate Parquet file
    df.to_parquet(
        os.path.join(out, f'training_nbac_part_{i}.parquet'),
        engine='pyarrow',
        compression='snappy',
        index=False
    )

# Record the end time
end_time = time.time()

# Calculate the elapsed time in seconds
elapsed_time_seconds = end_time - start_time

# Convert seconds to minutes
elapsed_time_minutes = elapsed_time_seconds / 60

print(f"NBAC Elapsed time: {elapsed_time_minutes:.2f} minutes")
