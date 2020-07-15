# UNET-Implementation
This is an implementation of U-Net architecture developed by [U-Net: Convolution Networks for Biomedical Image Segmentation] (https://arxiv.org/pdf/1505.04597.pdf).
---
Dataset used for testing this architecture is referred from https://github.com/zhixuhao.
Images are in input folder and data is prepared using dataset.py file.
Data augmentation is done using keras ImageDataGenerator as only 30 images were available.
I developed this arhictecture using tensorflow and keras.
I trained this model for 5 epochs and used 1000 steps per epoch.
I got the accuracy of 95%.
I have included the prediction in results folder.
