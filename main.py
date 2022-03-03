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
from model import build_unet
from data_gen import DataGenerator
from metrics import dice_coef, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing, precision, sensitivity, specificity

# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt 
import gif_your_nifti.core as gif2nif

# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
from keras import utils as np_utils
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

VOLUME_SLICES = 100 
VOLUME_START_AT = 22
IMG_SIZE = 128

TRAIN_DATASET_PATH = 'C:/Users/samue/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = 'C:/Users/samue/BraTS2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

input_layer = Input((IMG_SIZE, IMG_SIZE, 2))

model = build_unet(input_layer, 'he_normal', 0.2)
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001) )

# lists of directories with studies
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

# file BraTS20_Training_355 has ill formatted name for for seg.nii file
train_and_val_directories.remove(TRAIN_DATASET_PATH+'BraTS20_Training_355')


def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories); 

    
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2, shuffle=False) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15, shuffle=False)
np.save('history_all_fold.npy',test_ids)

training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

csv_logger = CSVLogger('training.log', separator=',', append=False)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1)
    ]

K.clear_session()

############ train model ################
#history =  model.fit(training_generator,
#                    epochs=35,
#                    steps_per_epoch=len(train_ids),
#                    callbacks= callbacks,
#                    validation_data = valid_generator
#                    )  
#np.save('history_all_fold.npy',history.history)
#model.save("model_x1_1.h5")

############ load trained model ################
model = keras.models.load_model('C:/Users/samue/bratsUnet/model_per_class.h5', 
                                   custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": dice_coef,
                                                   "precision": precision,
                                                   "sensitivity":sensitivity,
                                                   "specificity":specificity,
                                                   "dice_coef_necrotic": dice_coef_necrotic,
                                                   "dice_coef_edema": dice_coef_edema,
                                                   "dice_coef_enhancing": dice_coef_enhancing
                                                   }, compile=False)



# history = pd.read_csv('C:/Users/samue/bratsUnet/training_per_class.log', sep=',', engine='python')
# hist=history
# acc=hist['accuracy']
# val_acc=hist['val_accuracy']
# epoch=range(len(acc))
# loss=hist['loss']
# val_loss=hist['val_loss']
# train_dice=hist['dice_coef']
# val_dice=hist['val_dice_coef']
# f,ax=plt.subplots(1,4,figsize=(16,8))
# ax[0].plot(epoch,acc,'b',label='Training Accuracy')
# ax[0].plot(epoch,val_acc,'r',label='Validation Accuracy')
# ax[0].legend()
# ax[1].plot(epoch,loss,'b',label='Training Loss')
# ax[1].plot(epoch,val_loss,'r',label='Validation Loss')
# ax[1].legend()
# ax[2].plot(epoch,train_dice,'b',label='Training dice coef')
# ax[2].plot(epoch,val_dice,'r',label='Validation dice coef')
# ax[2].legend()
# ax[3].plot(epoch,hist['mean_io_u'],'b',label='Training mean IOU')
# ax[3].plot(epoch,hist['val_mean_io_u'],'r',label='Validation mean IOU')
# ax[3].legend()
# plt.show()

def predictByPath(case_path,case):
    files = next(os.walk(case_path))[2]
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii');
    flair=nib.load(vol_path).get_fdata()
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii');
    ce=nib.load(vol_path).get_fdata()  
    
    for j in range(VOLUME_SLICES):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        X[j,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        
    return model.predict(X/np.max(X), verbose=1)


def showPredictsById(case, start_slice = 56):
    path = f"C:/Users/samue/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    p = predictByPath(path,case)

    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1,6, figsize = (18, 50)) 

    for i in range(6): # for each image, add brain background
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
    
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    curr_gt=cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].title.set_text('all classes')
    axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[4].imshow(core[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    plt.show()

showPredictsById(case=test_ids[12][-3:])

#model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing] )
## Evaluate the model on the test data using `evaluate`
#print("Evaluate on test data")
#results = model.evaluate(test_generator, batch_size=60, callbacks= callbacks)
#print("test loss, test acc:", results)
