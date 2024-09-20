# !/usr/bin/env python
# coding: utf-8

# Read in packages

# In[21]:

from __future__ import division
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
from tensorflow.keras.metrics import Precision, Recall
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
import tensorflow as tf
# import tensorflow_addons as tfa
import logging
import time

# Record the start time
start_time = time.time()



#Define custom IOU (Intersection over Union) metric
# def iou_score(y_true, y_pred):
#     def f(y_true, y_pred):
#         intersection = tf.reduce_sum(y_true * y_pred)
#         union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
#         return (intersection + 1e-5) / (union + 1e-5)
#     return tf.numpy_function(f, [y_true, y_pred], tf.float32)

# # Define custom F1 Score metric
# def f1_score(y_true, y_pred):
#     def f(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.round(y_pred)  # Assuming y_pred is probabilities
#         tp = tf.reduce_sum(y_true * y_pred)
#         precision = tp / (tf.reduce_sum(y_pred) + 1e-5)
#         recall = tp / (tf.reduce_sum(y_true) + 1e-5)
#         return 2 * precision * recall / (precision + recall + 1e-5)
#     return tf.numpy_function(f, [y_true, y_pred], tf.float32)


min_max = pd.read_csv("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_global_min_max_cutoff_proj.csv").reset_index(drop = True)

min_max = min_max[['6', '7', '8']]

print(min_max)
#functin to standardize all bands at once


#function to standardize
def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

#function to normalize
def normalize(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    minv = np.min(a, axis=axis, keepdims=True)
    maxv = np.max(a, axis=axis, keepdims=True)
    return (a - minv) / (maxv - minv)


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
training_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/mtbs_training_files.csv')['Files'].tolist()
validation_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/mtbs_validation_files.csv')['Files'].tolist()
testing_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/mtbs_testing_files.csv')['Files'].tolist()

training_names2 = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/nbac_training_files.csv')['Files'].tolist()
validation_names2 = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/nbac_validation_files.csv')['Files'].tolist()
testing_names2 = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/nbac_testing_files.csv')['Files'].tolist()


training_names = training_names + training_names2 
validation_names = validation_names + validation_names2
testing_names = testing_names + testing_names2

# training_names = training_names[:10000]
# validation_names = validation_names[:10000]
# testing_names = testing_names[:10000]


from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()

#function to normalize within range
def normalize(start, end, arr):
    width = end - start
    res = (arr - np.nanmin(arr))/(np.nanmax(arr)- np.nanmin(arr)) * width + start

#     res = (arr - arr.min())/(arr.max() - arr.min()) * width + start
    return res

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
		
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32") #create matrix of zeros which will have the dimension height, wideth, n_bands), 8 is the n_bands
        
  
         #start populating x by enumerating over the input img paths
        for j, path in enumerate(batch_img_paths):

            #load image
            img =  np.round(np.load(path), 3)
            
            if img.shape[2] == 4:
                
                img = img[:, :, :-1]

                
            else:
                
                img = img[:, :, 6:9]

            # img = img * 1000
            img = img.astype(float)
            img = np.round(img, 3)
            img[img == 0] = -999

            img[np.isnan(img)] = -999


            img[img == -999] = np.nan

            in_shape = img.shape
            
            #turn to dataframe to normalize
            img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
			
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

            # img =  tf.keras.utils.to_categorical(img, num_classes = 2)
            # y[j] = np.expand_dims(img, 2) 
            y[j] = img
  
       
    #Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
    # y[j] -= 1

        return x, y


# Read in the images based on the generator

# In[24]:


#batch size and img size
BATCH_SIZE = 64
GPUS = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
strategy = tensorflow.distribute.MirroredStrategy() #can add GPUS here to select specific ones
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync



#image size
img_size = (128, 128)
# img_size = (128, 128)

#number of classes to predict
num_classes = 1

#get images
train_gen = img_gen(batch_size, img_size, training_names)
val_gen = img_gen(batch_size, img_size, validation_names)
test_gen = img_gen(batch_size, img_size, testing_names)
#

# Free up RAM in case the model definition cells were run multiple times
tensorflow.keras.backend.clear_session()

LR = 1e-4


optimizer = tensorflow.keras.optimizers.Adam(learning_rate=LR) #this is 1e-3, default or 'rmsprop'
    
# loss= tensorflow.keras.losses.BinaryFocalCrossentropy(
#     from_logits=False,
#     gamma = 2.0,
#     alpha = 0.25)

loss= tensorflow.keras.losses.BinaryFocalCrossentropy(
    from_logits=False)


# f.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25)
# loss = tensorflow.keras.losses.BinaryFocalCrossentropy(
#     apply_class_balancing=False,
#     alpha=0.25,
#     gamma=2.0,
#     from_logits=False,
#     label_smoothing=0.0,
#     axis=-1,
#     reduction=losses_utils.ReductionV2.AUTO,
#     name='binary_focal_crossentropy'
# )



callbacks = [tensorflow.keras.callbacks.ModelCheckpoint(
    filepath="/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/models/nbac_mtbs_swin_w4",
#     verbose=1,
    save_weights_only=False,
    save_best_only=True,
    monitor='val_iou_score',
    mode = 'max'),
    tensorflow.keras.callbacks.EarlyStopping(monitor='val_iou_score', mode = 'max',  patience=20),
    tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)]


# Open a strategy scope.
with strategy.scope():
  

    #one [16,32,64,128]
    #two [16,32,64,128,256]
    #three [32,64,128,256]
    #four [32,64,128,256,512]
    #five [16,32,64,128,256,512,1024]


    # model_unet_from_scratch = models.unet_plus_2d((None, None, 3), filter_num= [16,32,64,128], #make smaller64, 128, 256, 512,[16, 32, 64, 128]
    #                    n_labels=num_classes, 
    #                    stack_num_down=2, stack_num_up=2, 
    #                    activation='ReLU', 
    #                    output_activation='Sigmoid', 
    #                    batch_norm=True, pool=False, unpool=False, 
    #                    backbone='EfficientNetB7', weights=None, 
    #                    freeze_backbone=False, freeze_batch_norm=False, 
    #                    deep_supervision = True,
    #                    name='unet')

    #smaller patch size for more local detail (smaller fires?) and larger for more global
    model_unet_from_scratch = models.swin_unet_2d(
        input_size=(128, 128, 3),  # Allows variable input sizes
        filter_num_begin=64, 
        n_labels=1, 
        depth=4, 
        stack_num_down=2, 
        stack_num_up=2,
        # patch_size=(2, 2),  # Medium patch size for balanced detail and context was (4,4)
        # num_heads=[4, 8, 8, 8],  # More heads for capturing complex patterns
        # window_size=[4, 2, 2, 2],  # Smaller windows in early layers for detail, larger in deeper layers for context
        patch_size=(4, 4),  # Medium patch size for balanced detail and context was (4,4)
        num_heads=[8, 16, 16, 16],  # More heads for capturing complex patterns
        window_size=[8, 4, 4, 4],  # Smaller windows in early layers for detail, larger in deeper layers for context
        num_mlp=512, 
        output_activation='Sigmoid', 
        shift_window=True, 
        name='swin_unet'
    )

    # model_unet_from_scratch = models.unet_3plus_2d((None, None, 1), n_labels=num_classes, filter_num_down=[16,32,64,128], 
    #                          filter_num_skip='auto', filter_num_aggregate='auto', 
    #                         backbone='EfficientNetB7', weights=None, 
    #                          freeze_backbone=False,
    #                          stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
    #                          batch_norm=True, pool='max', unpool=False, deep_supervision=True, name='unet')
	
#     model.set_weights(listOfNumpyArrays)
    model_unet_from_scratch.compile(loss='binary_crossentropy',
                                    # loss = loss,
                                    # optimizer='adam',
                                    optimizer = optimizer,
                                    metrics=[sm.metrics.Precision(threshold=0.5),
                                      sm.metrics.Recall(threshold=0.5),
                                      sm.metrics.FScore(threshold=0.5), 
                                      sm.metrics.IOUScore(threshold=0.5),
                                      'accuracy'])
    
    # model_unet_from_scratch.compile(
    #                 optimizer='adam',
    #                 loss='binary_crossentropy',
    #                 metrics=[
    #                         Precision(name='precision'),
    #                         Recall(name='recall'),
    #                         f1_score,
    #                         iou_score,
    #                         'accuracy'
    #                     ]
    #                 )

#fit the model
history = model_unet_from_scratch.fit(
    train_gen,
    epochs=100,
    callbacks = callbacks,
    validation_data=val_gen,
    verbose = 0) 

# model_unet_from_scratch.save("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_079_128.h5")
model_unet_from_scratch.save("/explore/nobackup/people/spotter5/cnn_mapping/nbac_traiing/models/nbac_mtbs_swin_w4.tf")


history_dict = history.history

#save output
# result = pd.DataFrame({'Precision': history_dict["precision"],
#                        'Val_Precision': history_dict['val_precision'],
#                        'Recall': history_dict["recall"],
#                        'Val_Recall': history_dict['recall'],
#                        'F1': history_dict["f1-score"],
#                        'Val_F1': history_dict['val_f1-score'],
#                        'IOU': history_dict["iou_score"],
#                        'Val_IOU': history_dict['val_iou_score'],
#                        'Loss': history_dict['loss'],
#                        'Val_Loss': history_dict['val_loss']})

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
                      'Loss': history_dict['accuracy'],
                       'Val_Loss': history_dict['val_accuracy']})

# result = pd.DataFrame({'Precision': history_dict["unet_output_final_activation_precision"],
#                        'Val_Precision': history_dict['val_unet_output_final_activation_precision'],
#                        'Recall': history_dict["unet_output_final_activation_recall"],
#                        'Val_Recall': history_dict['val_unet_output_final_activation_recall'],
#                        'F1': history_dict["unet_output_final_activation_f1-score"],
#                        'Val_F1': history_dict['val_unet_output_final_activation_f1-score'],
#                        'IOU': history_dict["unet_output_final_activation_iou_score"],
#                        'Val_IOU': history_dict['val_unet_output_final_activation_iou_score'],
#                        'Loss': history_dict['unet_output_final_activation_loss'],
#                        'Val_Loss': history_dict['val_unet_output_final_activation_loss'],
#                       'Accuracy': history_dict['unet_output_final_activation_accuracy'],
#                        'Val_Accuracy': history_dict['val_unet_output_final_activation_accuracy']})


result.to_csv("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/nbac_mtbs_swin_w4.csv")



# Record the end time
end_time = time.time()

# Calculate the time difference in seconds
time_difference_seconds = end_time - start_time

# Convert seconds to hours
time_difference_hours = time_difference_seconds / 3600  # 1 hour = 3600 seconds

print(f"Time taken: {time_difference_hours:.2f} hours")