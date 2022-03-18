# Brats

This repository details a U-Net approach to solving the BRaTS Dataset from 2020, using a combination of specific hyperparameters and binary crossentropy for multi class segmentation. 

The BRaTS Dataset comprises of 220 high grade gliomas (HGG) and 54 low grade gliomas (LGG) MRIs. The four MRI modalities are T1, T1c, T2, and T2FLAIR. Segmented “ground truth” is 3 intra-tumoral classes. edema, enhancing tumor, and necrosis.


## U-Net architecture 
![68747470733a2f2f6c6d622e696e666f726d6174696b2e756e692d66726569627572672e64652f70656f706c652f726f6e6e656265722f752d6e65742f752d6e65742d6172636869746563747572652e706e67](https://user-images.githubusercontent.com/101694383/158929757-79641461-bd4b-4126-aa41-f909e831aa60.png)
