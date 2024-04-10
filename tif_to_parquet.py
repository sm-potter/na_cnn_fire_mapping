import rioxarray 
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import time
import random

# Record the start time
start_time = time.time()

#don't use files if all 0's

#check if all 0
def is_matrix_all_zeros(matrix):
    # Convert the matrix to a NumPy array
    np_matrix = np.array(matrix)

    # Check if all elements in the array are zeros
    return np.all(np_matrix == 0)

#outpath
out = '/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/parquet_files'
os.makedirs(out, exist_ok = True)


#code to read in all training data
all_files_mtbs =  glob.glob('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj_mtbs/*.tif')

all_files_nbac =  glob.glob('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj/*.tif')

#too many files for nbac, need to sample\
#50% sample size
sample_size = len(all_files_nbac) // 2

all_files_nbac = random.sample(all_files_nbac, sample_size)

# #empty list for combining all data
# combined_training = []

# #start with mtbs
# for f in all_files_mtbs:
    
#     print(f)

#     #read in file and convert to numpy
#     in_file = rioxarray.open_rasterio(os.path.join('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj_mtbs', f)).to_numpy()
    
#     #convert to band last
#     in_file= np.moveaxis(in_file, 0, 2) 

#     x = in_file[:, :, :-1]
#     x = x.astype(float)
#     x[x == 0] = np.nan
    
#     x = np.round(x, 2)
    
#     y = in_file[:, :, -1]
#     y = y.astype(float)
#     y[y <0 ] = 0
#     y[y >1 ] = 0
    
#     y[~np.isin(y, [0,1])] = np.nan
    
#     y = np.round(y, 2)
    
#     stacked = np.dstack([x, y])
    
#     #reshape the 3D matrix to 2D
#     x, y, z = in_file.shape  
    
#      #convert to pandas dataframe
#     reshaped_data = stacked.reshape(x*y, z)

#     band_names = ['blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2', 'dNBR', 'dNDVI', 'dNDII', 'y']

#     # Create a DataFrame
#     training = pd.DataFrame(reshaped_data, columns=band_names)

# #     #original values were originally scaled
# #     columns_to_divide = [col for col in training.columns if col != 'y']

# #     # Divide selected columns by 1000
# #     training[columns_to_divide] = training[columns_to_divide].div(1000).round(3)
    
# #     # training['Fname'] = f.replace('.tif', '')
    
# #     # training = training[['Fname', 'dNBR', 'dNDVI', 'dNDII', 'y']]
# #     training= training[~(training == 0).all(axis=1)]
# #     training = training[training['y'].isin([0, 1])]
    
#     #append to list
#     combined_training.append(training)
    
  
# #concat
# combined_training = pd.concat(combined_training, ignore_index=True)#.dropna()

# # combined_training.head()

# # Record the end time
# end_time = time.time()

# # Calculate the elapsed time in seconds
# elapsed_time_seconds = end_time - start_time


# combined_training.to_parquet(os.path.join(out, 'all_training_mtbs.parquet'), index = False)

# Convert seconds to minutes
# elapsed_time_minutes = elapsed_time_seconds / 60

# print(f"MTBS Elapsed time: {elapsed_time_minutes:.2f} minutes")
    
    
start_time = time.time()

#now do nbac
#empty list for combining all data
combined_training = []

#start with mtbs
for f in all_files_nbac:
    
    # print(f)
    try:
        #read in file and convert to numpy
        in_file = rioxarray.open_rasterio(os.path.join('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj_nbac', f)).to_numpy().astype(np.int16)

        #convert to band last
        in_file= np.moveaxis(in_file, 0, 2) 

        x = in_file[:, :, :-1]
        # x = x.astype(float)
        x[x == 0] = np.nan

        x = np.round(x, 2)

        y = in_file[:, :, -1]
        y = y.astype(float)
        y[y <0 ] = 0
        y[y >1 ] = 0

        # y[~np.isin(y, [0,1])] = np.nan

#         y = np.round(y, 2)
        stacked = np.dstack([x, y])

        #reshape the 3D matrix to 2D
        x, y, z = in_file.shape  # Get the 'x' dimension
        # matrix_2d = matrix_3d.reshape(x*x, 10)

         #convert to pandas dataframe
        reshaped_data = stacked.reshape(x*y, z)

        band_names = ['blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2', 'dNBR', 'dNDVI', 'dNDII', 'y']

        # Create a DataFrame
        training = pd.DataFrame(reshaped_data, columns=band_names)

    #     #original values were originally scaled
    #     columns_to_divide = [col for col in training.columns if col != 'y']

    #     # Divide selected columns by 1000
    #     training[columns_to_divide] = training[columns_to_divide].div(1000).round(3)

    #     # training['Fname'] = f.replace('.tif', '')

    #     # training = training[['Fname', 'dNBR', 'dNDVI', 'dNDII', 'y']]
    #     training= training[~(training == 0).all(axis=1)]
    #     training = training[training['y'].isin([0, 1])]

        #append to list
        combined_training.append(training)
        
    except:
        pass

  
#concat
combined_training = pd.concat(combined_training, ignore_index=True)#.dropna()

# combined_training.head()

# Record the end time
end_time = time.time()

# Calculate the elapsed time in seconds
elapsed_time_seconds = end_time - start_time


combined_training.to_parquet(os.path.join(out, 'all_training_nbac.parquet'), index = False)

# Convert seconds to minutes
elapsed_time_minutes = elapsed_time_seconds / 60

print(f"MTBS Elapsed time: {elapsed_time_minutes:.2f} minutes")
    
    
