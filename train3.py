import os
import numpy as np
from data_generator import imageLoader
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random
from unet_arch import unet_model
import pandas as pd
import segmentation_models_3D as sm
from keras.metrics import MeanIoU
from matplotlib import pyplot as plt
import random
from optparse import OptionParser
import tensorflow_addons as tfa

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.chdir('C:/Users/samue/UNET - pretrained/')

train_img_dir = "C:/Users/samue/BraTS2020/BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "C:/Users/samue/BraTS2020/BraTS2020_TrainingData/input_data_128/train/masks/"

val_img_dir = "C:/Users/samue/BraTS2020/BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "C:/Users/samue/BraTS2020/BraTS2020_TrainingData/input_data_128/val/masks/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list,
                              val_mask_dir, val_mask_list, batch_size)

img, msk = train_img_datagen.__next__()

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)

wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

model = unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_DEPTH=128,
                          IMG_CHANNELS=3,
                          num_classes=4)

model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

from keras.models import load_model

my_model = load_model('brats_3d.hdf5',
                      compile=False)

batch_size = 5  # Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list,
                               val_mask_dir, val_mask_list, batch_size)

test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

#val_loss, val_dice = model.evaluate(val_img_datagen)
#print(f"validation soft dice loss: {val_loss:.4f}")
#print(f"validation dice coefficient: {val_dice:.4f}")

#n_classes = 4
#IOU_keras = MeanIoU(num_classes=n_classes)
#IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
#print("Mean IoU =", IOU_keras.result().numpy())

#brain val
val_num = 12
test_val = np.load("C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\input_data_3channels\\images\\image_" + str(val_num) + ".npy")

test_val_img_input = np.expand_dims(test_val, axis=0)
test_val_prediction = my_model.predict(test_val_img_input)
test_val_prediction_argmax = np.argmax(test_val_prediction, axis=4)[0, :, :, :]

n_slice_val=70

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Validation Test Image')
plt.imshow(test_val[:, :, n_slice_val, 1], cmap='gray')
plt.subplot(232)
plt.title('Prediction on validation image')
plt.imshow(test_val_prediction_argmax[:, :, n_slice_val])
plt.show()


#liver TEST val###############################

test_liver = np.load("D:\\MDS\\dataset\\Task03_Liver\\numpy\\val\\liverTEST_3.npy")
liver_mask = np.load("D:\\MDS\\dataset\\Task03_Liver\\numpy\\val\\liverTEST_3.npy")

test_val_liver_input = np.expand_dims(test_liver, axis=0)
test_val_liver_prediction = my_model.predict(test_val_liver_input)
test_val_liver_prediction_argmax = np.argmax(test_val_liver_prediction, axis=4)[0, :, :, :]
test_mask_argmax = np.argmax(test_mask, axis=1)

n_slice_liver=random.randint(90,128)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_liver[:, :, n_slice_liver, 1], cmap='gray')
plt.subplot(232)
plt.title('Prediction on test image')
plt.imshow(test_val_liver_prediction_argmax[:, :, n_slice_liver])
plt.show()
