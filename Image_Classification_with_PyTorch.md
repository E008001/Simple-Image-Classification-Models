## Image Classification with PyTorch
PyTorch, one of the newer Python-focused frameworks for designing deep learning workflows that can be easily productionized.  
PyTorch is one of many frameworks that have been designed for this purpose and work well with Python, among popular ones like TensorFlow and Keras. It also offers strong support for GPUs.  
(pic)[https://github.com/E008001/Simple-Image-Classification-Model/blob/master/con8.jpg]

we will build an image classification model,which will help to understand the shape of an image and the distribution of classes.   
to prepare data for optimum modeling results and then build a convolutional neural network (CNN) that will classify images. 
### Why Convolutional Neural Networks (CNNs)?
why we need CNNs in the first place and how they are helpful???  
**We can consider Convolutional Neural Networks, or CNNs, as feature extractors that help to extract features from images.**  
1- This is the problem with artificial neural networks – they lose spatial orientation.
### Large number of parameters
Another problem with neural networks is the large number of parameters.  
these parameters will only increase as we increase the number of hidden layers. So, the two major disadvantages of using artificial neural networks are:  
**Loses spatial orientation of the image  
The number of parameters increases drastically**   
So how do we deal with this problem? How can we preserve the spatial orientation as well as reduce the learnable parameters?  
This is where convolutional neural networks can be really helpful. CNNs help to extract features from the images which may be helpful
in classifying the objects in that image. It starts by extracting low dimensional features (like edges) from the image, and then some high dimensional features like the shapes.

**We use filters to extract features from the images and Pooling techniques to reduce the number of learnable parameters.**  
### Brief to Convolutional Neural Networks - CNNs
A CNN is primarily a stack of layers of convolutions, often interleaved with normalization and activation layers. The components of a convolutional neural network is:  
CNN — A stack of convolution layers  
Convolution Layer — A layer to detect certain features. Has a specific number of channels.  
Channels — Detects a specific feature in the image.  
Kernel/Filter — The feature to be detected in each channel. It has a fixed size, usually 3 x 3.  
### Importing Library  
To begin, import the **torch** and **torchvision** frameworks and their libraries with **numpy**, **pandas**, and **sklearn**. Libraries and functions used in the code below include:
transforms, for basic image transformations
**torch.nn.functional**, which contains useful activation functions
Dataset and Dataloader, PyTorch's data loading utility
```
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
```
### Defining the Model Structure  
Models are defined in PyTorch by custom classes that extend the Module class. All the components of the models can be found in the torch.nn package.
### loading dataset  


load data

### Creating a validation set and preprocessing the images
```
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
```

