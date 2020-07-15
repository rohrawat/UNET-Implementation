# UNET-Implementation

This is an implementation of U-Net architecture developed by [U-Net: Convolution Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf).
---
Dataset used for testing this architecture is referred from https://github.com/zhixuhao.<br>
Images are in input folder and data is prepared using dataset.py file.<br>
Data augmentation is done using keras ImageDataGenerator as only 30 images were available.<br>
I developed this arhictecture using tensorflow and keras.<br>
I trained this model for 5 epochs and used 1000 steps per epoch.<br>
I got the accuracy of 95%.<br>
I have included the prediction in results folder.<br>
UNET.py file is the exact replica of architecture as the input in architecture is 572 * 572 image and output is 388 * 388 image.
