# Brats

This repository details a U-Net approach to solving the BRaTS Dataset from 2020, using a combination of specific hyperparameters and binary crossentropy for multi class segmentation. 

The BRaTS Dataset comprises of 220 high grade gliomas (HGG) and 54 low grade gliomas (LGG) MRIs. The four MRI modalities are T1, T1c, T2, and T2FLAIR. Segmented “ground truth” is 3 intra-tumoral classes. edema, enhancing tumor, and necrosis.

For each file of the framework, it was loaded as a np array using 

The framework evaluation results are calculated using Keras submodules and set of  custom metrics calculating the Mean Intersection over Union, the [Dice Coefficient](https://chenriang.me/f1-equal-dice-coefficient.html) of the overall preidiciton and each class. It was evaluated over 32 images in a batch size of 10:

```python
Metric Evaluation
32/32 [==============================] - 11s 241ms/step - loss: 0.2069 - accuracy: 0.9924 - mean_io_u_1: 0.8061 - dice_coef: 0.5741 - precision: 0.9937 - sensitivity: 0.9902 - specificity: 0.9978 - dice_necrotic: 0.4602 - dice_edema: 0.7037 - dice_enhancing: 0.6893

```

## U-Net architecture 
![68747470733a2f2f6c6d622e696e666f726d6174696b2e756e692d66726569627572672e64652f70656f706c652f726f6e6e656265722f752d6e65742f752d6e65742d6172636869746563747572652e706e67](https://user-images.githubusercontent.com/101694383/158929757-79641461-bd4b-4126-aa41-f909e831aa60.png)

