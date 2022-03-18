import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import torchio as tio

scaler = MinMaxScaler()

TRAIN_DATASET_PATH = "C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\"
VAL_DATASET_PATH = "C:\\Users\\samue\\BraTS2020\\BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData\\"
LIVER_DATASET_PATH = "D:\\MDS\\dataset\\Task03_Liver\\"

#Training set
t2_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\*\\*t2.nii'))
t1ce_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\*\\*t1ce.nii'))
flair_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\*\\*flair.nii'))
mask_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\*\\*seg.nii'))

#Validation Set
val_t2_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData\\*\\*t2.nii'))
val_t1ce_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData\\*\\*t1ce.nii'))
val_flair_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData\\*\\*flair.nii'))


#Training numpy conversion
for img in range(len(t2_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)
      
    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
    
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
   
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val, counts = np.unique(temp_mask, return_counts=True)

    temp_mask= to_categorical(temp_mask, num_classes=4)
    np.save('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\input_data_3channels\\images\\image_'+str(img)+'.npy', temp_combined_images)
    np.save('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\input_data_3channels\\masks\\mask_'+str(img)+'.npy', temp_mask)

#Validation numpy conversion
for img2 in range(len(val_t2_list)):  
    print("Now preparing image number: ", img2)
      
    val_image_t2=nib.load(val_t2_list[img2]).get_fdata()
    val_image_t2=scaler.fit_transform(val_image_t2.reshape(-1, val_image_t2.shape[-1])).reshape(val_image_t2.shape)
   
    val_image_t1ce=nib.load(val_t1ce_list[img2]).get_fdata()
    val_image_t1ce=scaler.fit_transform(val_image_t1ce.reshape(-1, val_image_t1ce.shape[-1])).reshape(val_image_t1ce.shape)
   
    val_image_flair=nib.load(flair_list[img2]).get_fdata()
    val_image_flair=scaler.fit_transform(val_image_flair.reshape(-1, val_image_flair.shape[-1])).reshape(val_image_flair.shape)
        
    temp_combined_val_images = np.stack([val_image_flair, val_image_t1ce, val_image_t2], axis=3)
   
    temp_combined_val_images = temp_combined_val_images[56:184, 56:184, 13:141]
    
    np.save('C:\\Users\\samue\\BraTS2020\\BraTS2020_ValidationData\\input_val_images\\val\\image_'+str(img2)+'.npy', temp_combined_val_images)

#Liver Validation set 
val_liver_list = sorted(glob.glob('D:\\MDS\\dataset\\Task03_Liver\\imagesTr\\*.nii'))
mask_liver_list = sorted(glob.glob('D:\\MDS\\dataset\\Task03_Liver\\labelsTr\\*.nii'))

###LIVER TEST 1 PARSE####

for liverimg in range(0,4):  
    print("Now preparing image number: ", liverimg)
      
    
    val_liver=nib.load(val_liver_list[liverimg]).get_fdata()
    val_liver=scaler.fit_transform(val_liver.reshape(-1, val_liver.shape[-1])).reshape(val_liver.shape)
       
    val_mask=nib.load(mask_liver_list[liverimg]).get_fdata()
    val_mask=val_mask.astype(np.uint8)
    
    import skimage.transform as skTrans
    result1 = skTrans.resize(val_liver, (128,128,512), order=1, preserve_range=True)  
    
    temp_combined_val_images = np.stack([result1, result1, result1], axis=3)
    
    np.save('D:\\MDS\\dataset\\Task03_Liver\\numpy\\val\\liverTEST3_'+str(liverimg)+'.npy', temp_combined_val_images)
    np.save('D:\\MDS\\dataset\\Task03_Liver\\numpy\\mask\\liverTEST3_'+str(liverimg)+'.npy', val_mask)
