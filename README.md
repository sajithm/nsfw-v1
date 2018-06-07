# nsfw-v1
A nudity detector developed with Python

## Introduction
A nudity detector developed using Keras. It uses 3 convolution layers with maxpool, followed by a fully connected layer and sigmoid at the end.

## Dataset
The dataset is not included in the repository. It can be downloaded from [here](https://www.dropbox.com/s/t3fbvk43nmvwgfy/dataset.zip?dl=0).

The test/train images were resized to fit in 0.3MP (640x480 or less). Training set contains 4000 images each in SFW and NSFW categories while the test set contains 500 images each.

Binary model is included in the repo. If you wish to train with your own data, it is recommended that you use a larger dataset.
