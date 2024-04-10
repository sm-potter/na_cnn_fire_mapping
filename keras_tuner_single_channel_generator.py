#!/usr/bin/env python
# coding: utf-8
#Read in packages
# In[1]:


#bayesian optimzation for one band. 

import pandas as pd
import tensorflow
import os
import shutil
from tensorflow import keras
import tensorflow as tf
import segmentation_models as sm
# import matplotlib.pyplot as plt
import numpy as np
from keras_unet_collection import models
import tensorflow_addons as tfa
import optuna
from optkeras.optkeras import OptKeras
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
import kerastuner as kt
import random


#theoretically we would combine mtbs and nbac but it is the same anyways
min_max = pd.read_csv("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_global_min_max_cutoff_proj.csv").reset_index(drop = True)

min_max = min_max[['6']]


#function to get files from storage bucket
def get_files(bucket_path):

	"""argument is the path to where the numpy
	save files are located, return a list of filenames
	"""
	all = []

	#list of files
	files = os.listdir(bucket_path)

	#get list of filenames we will use, notte I remove images that don't have a target due to clouds
	file_names = []
	for f in files:

		if f.endswith('.npy'):


			all.append(os.path.join(bucket_path, f))
	return(all)


#get all the pathways
training_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj_0_final_128_training_files.csv')['Files'].tolist()
validation_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj_0_final_128_validation_files.csv')['Files'].tolist()
testing_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj_0_final_128_testing_files.csv')['Files'].tolist()

training_names2 = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj_mtbs_0_128_training_files.csv')['Files'].tolist()
validation_names2 = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj_mtbs_0_128_validation_files.csv')['Files'].tolist()
testing_names2 = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_proj_mtbs_0_128_testing_files.csv')['Files'].tolist()

training_names = training_names + training_names2

#Randomly sample 50% of the elements
sample_size = int(len(training_names) * 0.3)
training_names = random.sample(training_names, sample_size)

validation_names = validation_names + validation_names2
testing_names = testing_names + testing_names2


from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()

class img_gen(tensorflow.keras.utils.Sequence):

    """Helper to iterate over the data (as Numpy arrays).
    Inputs are batch size, the image size, the input paths (x) and target paths (y)
    """

    #will need pre defined variables batch_size, img_size, input_img_paths and target_img_paths
    def __init__(self, batch_size, img_size, input_img_paths):
	    self.batch_size = batch_size
	    self.img_size = img_size
	    self.input_img_paths = input_img_paths
	    self.target_img_paths = input_img_paths

    #number of batches the generator is supposed to produceis the length of the paths divided by the batch siize
    def __len__(self):
	    return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_img_paths = self.input_img_paths[i : i + self.batch_size] #for a given index get the input batch pathways (x)
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size] #for a given index get the input batch pathways (y)
		
        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32") #create matrix of zeros which will have the dimension (batch_size, height, wideth, n_bands), 8 is the n_bands
        
  
        #start populating x by enumerating over the input img paths
        for j, path in enumerate(batch_img_paths):

            #load image
            img =  np.round(np.load(path), 3)[:, :, 6]

            # img = img * 1000
            img = img.astype(float)
            img = np.round(img, 3)
            img[img == 0] = -999

            img[np.isnan(img)] = -999


            img[img == -999] = np.nan

            in_shape = img.shape

            #turn to dataframe to normalize
            img = img.reshape(img.shape[0] * img.shape[1])

            img = pd.DataFrame(img)

            img.columns = min_max.columns

            img = pd.concat([min_max, img]).reset_index(drop = True)


            #normalize 0 to 1
            img = pd.DataFrame(scaler.fit_transform(img))

            img = img.iloc[2:]
#
#             img = img.values.reshape(in_shape)
            img = img.values.reshape(in_shape)

#             replace nan with -1
            img[np.isnan(img)] = -1

#apply standardization
# img = normalize(img, axis=(0,1))

            img = np.round(img, 3)
            #populate x
            x[j] = img#[:, :, 4:] index number is not included, 


        #do tthe same thing for y
        y = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")

        for j, path in enumerate(batch_target_img_paths):

            #load image
            img =  np.round(np.load(path), 3)[:, :, -1]

            img = img.astype(int)

            img[img < 0] = 0
            img[img >1] = 0
            img[~np.isin(img, [0,1])] = 0

            img[np.isnan(img)] = 0
            img = img.astype(int)

            y[j] = img
  
       

        return x, y





tensorflow.keras.backend.clear_session()


# In[4]:


input_height = 128
input_width = 128
num_channels = 1
n_labels = 1
hyperparameters = {
    'learning_rate': (1e-5, 1e-2, 'log-uniform'),
    'filters': [16, 32, 64],
}


def build_model(hp):
    
    #need for tuning filter numbers
    filter_num_options = [[16,32,64,128], [32,64,128,256]]
    filter_num_choice = hp.Choice('filter_num', values=[0, 1])  # To select between the two options
    selected_filter_num = filter_num_options[filter_num_choice]
        
    model = models.unet_plus_2d((None, None, num_channels),
                               filter_num=selected_filter_num,
                               # filter_num = [16,32,64,128],
                               activation=hp.Choice('activation', values=['ReLU', 'GELU', 'Snake']),
                               n_labels = n_labels,
                               stack_num_up = hp.Choice('stack_num_up', values = [1,2]),
                               stack_num_down =  hp.Choice('stack_num_down', values = [1,2]),
                               output_activation = 'Sigmoid',
                               batch_norm = True,
                               pool = hp.Choice('pool', values = [True,False]),
                               # unpool = hp.Choice('unpool', values = [True,False]),
                               unpool = False,
                               backbone = hp.Choice('backbone', values = ['EfficientNetB7', 'VGG19', 'ResNet152', 'DenseNet169']),
                               # backbone = 'EfficientNetB7'
                               weights = None,
                               deep_supervision = True
                               )
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model



#if directory already exists it will fail
directory = '/explore/nobackup/people/spotter5/cnn_mapping/keras_tuner_test'
if os.path.isdir(directory):
    shutil.rmtree(directory)

tuner = kt.BayesianOptimization(
    build_model,
    # objective= kt.Objective('val_mean_iou', direction="max"),  
    objective= 'val_loss',  

    max_trials=10,
    directory=directory,
    project_name='unet_tuning'
)

tuner.search_space_summary()

# Add the filter_num hyperparameter to the search space
# tuner.search_space.update({'filter_num': [16, 32, 64]})


# In[6]:


#batch size and img size
BATCH_SIZE = 15
GPUS = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
strategy = tensorflow.distribute.MirroredStrategy() #can add GPUS here to select specific ones
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

#image size
img_size = (128, 128)

num_classes = 1

train_gen = img_gen(batch_size, img_size, training_names)
val_gen = img_gen(batch_size, img_size, validation_names)
test_gen = img_gen(batch_size, img_size, testing_names)


# In[7]:


num_epochs = 30
tuner.search(train_gen, epochs=num_epochs, validation_data=val_gen)

# Get the best model hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)


# In[17]:


final = pd.DataFrame([best_hps.values])
final.to_csv(os.path.join(directory, 'mtbs_tuned.csv'), index = False)

