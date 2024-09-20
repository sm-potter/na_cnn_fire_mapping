# !/usr/bin/env python
# coding: utf-8

# Read in packages

# In[21]:


import pandas as pd
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.lib.io import file_io
from tensorflow.python.keras.optimizer_v2.adam import Adam
import os
import segmentation_models as sm
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, Conv2DTranspose, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Input, AvgPool2D
from tensorflow.keras.models import Model
from keras_unet_collection import models
# import tensorflow_addons as tfa
import logging
import time

# Record the start time
start_time = time.time()



#minimum and maximum values to apply same normalization during draining
min_max = pd.read_csv("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_global_min_max_cutoff_proj.csv").reset_index(drop = True)

#6, 7, 8 correspond to dNBR, dNDVI, dNDII, if you change these bands this must change. .
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


#get all the pathways for mtbs., 128x128 chunks
training_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/mtbs_training_files.csv')['Files'].tolist()
validation_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/mtbs_validation_files.csv')['Files'].tolist()
testing_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/mtbs_testing_files.csv')['Files'].tolist()

#get all pathways to nbac
training_names2 = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/nbac_training_files.csv')['Files'].tolist()
validation_names2 = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/nbac_validation_files.csv')['Files'].tolist()
testing_names2 = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/nbac_testing_files.csv')['Files'].tolist()

#combine
training_names = training_names + training_names2 
validation_names = validation_names + validation_names2
testing_names = testing_names + testing_names2

#scale 0-1
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()


#image generator for 3 bands, some pre processing here to turn nan to 0, ensure the min-max values are used for normalization.  
#the numpy arrays have bands red, green, blue, nir, swir1, swir2, dNBR, dNDVI, dNDII, y. 
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




#batch size and img size
BATCH_SIZE = 45
GPUS = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
strategy = tensorflow.distribute.MirroredStrategy() #can add GPUS here to select specific ones
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

#image size
img_size = (128, 128)

#number of classes to predict
num_classes = 1

#get images
train_gen = img_gen(batch_size, img_size, training_names)
val_gen = img_gen(batch_size, img_size, validation_names)
test_gen = img_gen(batch_size, img_size, testing_names)


# Free up RAM in case the model definition cells were run multiple times
tensorflow.keras.backend.clear_session()

#set learning rate
LR = 1e-3


optimizer = tensorflow.keras.optimizers.Adam(learning_rate=LR) #this is 1e-3, default or 'rmsprop'
    
#callbacks to save only best model, what to monitor for early stopping and reduce learning rate etc. 
callbacks = [tensorflow.keras.callbacks.ModelCheckpoint(
    filepath="/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/models/nbac_mtbs_regularize_50_global_norm_unet1_dnbr",
#     verbose=1,
    save_weights_only=False,
    save_best_only=True,
    monitor='val_iou_score',
    mode = 'max'),
    tensorflow.keras.callbacks.EarlyStopping(monitor='val_iou_score', mode = 'max',  patience=20),
    tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)]

#training loop
# Open a strategy scope, to use multiple gpus
with strategy.scope():

    model_unet_from_scratch = models.unet_2d((None, None, 1), filter_num= [16,32,64,128], #make smaller64, 128, 256, 512,[16, 32, 64, 128]
                       n_labels=num_classes, 
                       stack_num_down=2, stack_num_up=2, 
                       activation='ReLU', 
                       output_activation='Sigmoid', 
                       batch_norm=True, pool=False, unpool=False, 
                       backbone='EfficientNetB7', weights=None, 
                       freeze_backbone=False, freeze_batch_norm=False,
                       name='unet')

    model_unet_from_scratch.compile(loss='binary_crossentropy',
                                    optimizer='adam',
                                    metrics=[sm.metrics.Precision(threshold=0.5),
                                      sm.metrics.Recall(threshold=0.5),
                                      sm.metrics.FScore(threshold=0.5), 
                                      sm.metrics.IOUScore(threshold=0.5),
                                      'accuracy'])

#fit the model
history = model_unet_from_scratch.fit(
    train_gen,
    epochs=50,
    callbacks = callbacks,
    validation_data=val_gen,
    verbose = 0) 

#save final model
model_unet_from_scratch.save("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/models/nbac_mtbs_regularize_50_global_norm_unet1_dnbr.tf")


history_dict = history.history

#training/val results
result = pd.DataFrame({'Precision': history_dict["precision"],
                       'Val_Precision': history_dict['val_precision'],
                       'Recall': history_dict["recall"],
                       'Val_Recall': history_dict['recall'],
                       'F1': history_dict["f1-score"],
                       'Val_F1': history_dict['val_f1-score'],
                       'IOU': history_dict["iou_score"],
                       'Val_IOU': history_dict['val_iou_score'],
                       'Loss': history_dict['loss'],
                       'Val_Loss': history_dict['val_loss'],
                       'Accuracy': history_dict['accuracy'],
                       'Val_Accuracy': history_dict['val_accuracy']})

#save to csv
result.to_csv("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/nbac_mtbs_regularize_50_global_norm_unet1_dnbr.csv")



# Record the end time
end_time = time.time()

# Calculate the time difference in seconds
time_difference_seconds = end_time - start_time

# Convert seconds to hours
time_difference_hours = time_difference_seconds / 3600  # 1 hour = 3600 seconds

print(f"Time taken: {time_difference_hours:.2f} hours")