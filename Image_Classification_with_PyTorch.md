## Image Classification with PyTorch
PyTorch, one of the newer Python-focused frameworks for designing deep learning workflows that can be easily productionized.  
PyTorch is one of many frameworks that have been designed for this purpose and work well with Python, among popular ones like TensorFlow and Keras. It also offers strong support for GPUs.  
![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/con8.jpg)

we will build an image classification model,which will help to understand the shape of an image and the distribution of classes.   
to prepare data for optimum modeling results and then build a convolutional neural network (CNN) that will classify images.  
[10-model-deseas-planet](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/10-model-deseasplanet.ipynb)  
https://colab.research.google.com/drive/1hWgNBvqul3ZgH6JENxvGvdgakpxwD6wQ
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

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/CNNarchitec.jpg)  

A CNN is primarily a stack of layers of convolutions, often interleaved with normalization and activation layers.(a 2d Convolution Layer is an multiplication between the input and the different filters, where filters and inputs are 2d matrices) The components of a convolutional neural network is:   
CNN A stack of convolution layers  
Convolution Layer  A layer to detect certain features. Has a specific number of channels.  
Channels  Detects a specific feature in the image.  
Kernel/Filter  The feature to be detected in each channel. It has a fixed size, usually 3 x 3.  
  
![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/element-wise.jpg)  
```
first_conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
```
We use the Conv2d layer because our image data is two dimensional.  
**in_channels**:The value of in_channels equal to the number of channels in the layer above or in the case of the first layer, the number of channels in the data.  
**out_channels**: is the dimensionalityof the output space(the number of output filters in the convolution) larger number of out_channels allows the layer to potentially learn more useful features about the input data but the size of your CNN is a function of the number of in_channels/out_channels in each layer of your network and the number of layers. If you have a limited dataset, then you should aim to have a smaller network so that it can extract useful features from the data without overfitting.  
The **kernel_size** is the size of the filter that is run over the images.  
If we change the kernel_size to 5, the context would be expanded to include pixels adjacent to the central pixel.
The **stride** argument indicates how far the filter is moved after each computation.  
With a stride of 1 , a computation will be done for every pixel in the image.
With a stride of 2, every second pixel will have computation done on it, and the output data will have a height and width that is half the size of the input data.  
It is not recommended changing the stride from 1 without understanding of how this impacts the data moving through the network.  
The **padding** argument indicates how much 0 padding is added to the edges of the data during computation.  
The output size of any dimension from either a convolutional filtering or pooling operation can be calculated by the following equation:  
![image](https://github.com/E008001/Simple-Image-Classification-Models/blob/master/padding%20size.png)  
F is the filter size, P is the padding and S is the stride.  
If we wish to keep our input and output dimensions the same, with a filter size of 5 and a stride of 1, from the above formula we need a padding of 2.  
therfore the padding should be equal to the kernel size minus 1 divided by 2.
So for a kernel size of 3, we would have a padding of 1. In a kernel size of 5, we would have a padding of 2. In a kernel size of 7, we would have a padding of 3
This prevents the image shrinking as it moves through the layers.  
**why we use odd-size kernel and smaller kernel size mostly??**  
https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363  
https://icecreamlabs.com/2018/08/19/3x3-convolution-filters%E2%80%8A-%E2%80%8Aa-popular-choice/  
https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size  

### Pooling  layerts  
There are two benefits to using pooling in CNNs
It reduces the number of parameters by **down-sampling** and It makes feature detection more robust to object orientation and scale changes  
The most common type of pooling is called max pooling, and it applies the max function over the window  

Maxpool Layers (2x2 filter) is about taking the maximum element of a small (2x2) square that we delimitate from the input  
![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/maxpool.gif)  



### Common architectures in convolutional neural networks.  
AlexNet  
VGG 16   
U-Net  
GoogLeNet   
ResNet  
DenseNet  
https://pytorch.org/docs/stable/torchvision/models.html  

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/vgg16%20(1).png)  

VGG Net use in very Deep Convolutional Networks for Large Scale Image Recognition  

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/U-Net.png)  

U-Net used to medical image segmentation  

### Implementing Convolutional Neural Networks in PyTorch  
how to create Convolutional Neural Networks in PyTorch? 
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
import the required libraries
```
# importing the libraries
#import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
%matplotlib inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
```
### loading dataset  
```
# loading dataset
train=pd.read_csv('Book20.csv')

train.head(22)
```
```
train_img = []
for img_name in tqdm(train['Image_Name']):
    # defining the image path
    image_path =  str(img_name) + '.JPG'
    
    # reading the image
    img = imread(image_path, as_gray=True )
    img = resize(img, (img.shape[0] // 8, img.shape[1] // 8))
                      
    # normalizing the pixel values
    img /= 255.0
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    train_img.append(img)
    ```
    
    ```
    # converting the list to numpy array
train_x = np.array(train_img)
# defining the target
train_y = train['Lable'].values
train_x.shape
```
#### Creating a validation set 
we use 20% data in the validation set and the remaining in the training set.
```
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
```
**convert the images and the targets into torch format**:
```
# converting training images into torch format
train_x = train_x.reshape(16, 1, 28, 28)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int);
train_y = torch.from_numpy(train_y)

# shape of training data
train_x.shape, train_y.shape
```
convert the validation images:
```
# converting validation images into torch format
val_x = val_x.reshape(6000, 1, 28, 28)
val_x  = torch.from_numpy(val_x)
# converting the target into torch format
val_y = val_y.astype(int);
val_y = torch.from_numpy(val_y)
# shape of validation data
val_x.shape, val_y.shape
```


### Defining the Model Structure  
use a simple CNN architecture with 2 convolutional layers to extract features from the images.
then using a fully connected dense layer to classify those features into their categories.
```
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
```
define the optimizer and the loss function for the model:
```
	# defining the model
	model = Net()
	# defining the optimizer
	optimizer = Adam(model.parameters(), lr=0.07)
	# defining the loss function
	criterion = CrossEntropyLoss()
	# checking if GPU is available
	if torch.cuda.is_available():
	    model = model.cuda()
	    criterion = criterion.cuda()
	   
	print(model)
```
another model 
```
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 2)
	
   def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.drop_out(out)
    out = self.fc1(out)
    out = self.fc2(out)
    return out
```	
 Note that, after self.layer2, we apply a reshaping function to out, which flattens the data dimensions from 7 x 7 x 64 into 3164 x 1
```
model = CNN()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```


https://github.com/E008001/Simple-Image-Classification-Model/blob/master/torch-test-10.ipynb  
[an example of image classification in 2 classes](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/torch-test-10.ipynb)



![image](https://raw.githubusercontent.com/E008001/Simple-Image-Classification-Model/master/pattern.webp)

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/pattern.gif)

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/convolve.gif)

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/pattern2.png)

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/cnn-archit1.jpg)  

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/hidden1.png)

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/1D-convolutional-example_2x.png)

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/2D-convolution.png)

### 3D Convolutional Example  

![image](https://github.com/E008001/Simple-Image-Classification-Model/blob/master/3d-convolution.gif)


