import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  

# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt 
import gif_your_nifti.core as gif2nif


# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing
from keras import utils as np_utils

SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

SLICES = 100 
START_SLICE = 22
IMG_SIZE = 128

TRAIN_DATASET_PATH = 'C:/Users/samue/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = 'C:/Users/samue/BraTS2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size*SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*SLICES, 240, 240))
        Y = np.zeros((self.batch_size*SLICES, *self.dim, 4))

        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)
            
            flair_path = os.path.join(case_path, f'{i}_flair.nii');
            flair = nib.load(flair_path).get_fdata()    
            ce_path = os.path.join(case_path, f'{i}_t1ce.nii');
            ce = nib.load(ce_path).get_fdata()
            seg_path = os.path.join(case_path, f'{i}_seg.nii');
            seg = nib.load(seg_path).get_fdata()
            for j in range(SLICES):
                X[j +SLICES*c,:,:,0] = cv2.resize(flair[:,:,j+START_SLICE], (IMG_SIZE, IMG_SIZE));
                X[j +SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+START_SLICE], (IMG_SIZE, IMG_SIZE));
                y[j +SLICES*c] = seg[:,:,j+START_SLICE];
                    
        # Generate masks
        y[y==4] = 3
        mask = tf.one_hot(y, 4)                
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
        return X/np.max(X), Y
