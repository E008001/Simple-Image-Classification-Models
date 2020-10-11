## Image Classification with PyTorch
we will build an image classification model,which will help to understand the shape of an image and the distribution of classes.   
to prepare data for optimum modeling results and then build a convolutional neural network (CNN) that will classify images.  
### Importing Library and Data  
To begin, import the **torch** and **torchvision** frameworks and their libraries with **numpy**, **pandas**, and **sklearn**. Libraries and functions used in the code below include:
transforms, for basic image transformations
*torch.nn.functional*, which contains useful activation functions
Dataset and Dataloader, PyTorch's data loading utility
```
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
```
### Defining the Model Structure  
Models are defined in PyTorch by custom classes that extend the Module class. All the components of the models can be found in the torch.nn package.
