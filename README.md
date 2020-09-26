# Simple-Image-Classification-structure
Build your First Simple Image Classification Model 
### What is Image Classification in deep learning?  
Image classification is a supervised learning problem: define a set of target classes (objects to identify in images), and train a model to recognize them using labeled example photos,(Convolutional Neural Networks (CNNs) is the most popular neural network model being used for image classification problem.)
### Set the Structure of Image Data  
Our data needs to be in a particular format in order to solve an image classification problem.
You should have 2 folders, one for the train set and the other for the test set. In the training set, you will have a .csv file and an image folder: (in this case)  
•	The .csv file contains the names of all the training images and their corresponding true labels  
•	The image folder has all the training images.  
The .csv file in our test set is different from the one present in the training set. This test set .csv file contains the names of all the test images, but they do not have any corresponding labels. Our model will be trained on the images present in the training set and the label predictions will happen on the testing set images  
In the other case, **if our database does not have a .csv file**, after uploading the images, we will assign the labels separately to each of the folders containing different data classes based on the number of folders.
### the Process of Model Building  
How an image classification model is typically designed. 
.	Loading and pre-processing Data 
.	Defining Model architecture 
.	Training the model 
.	Estimation of performance  
### 1: Loading and pre-processing the data
we train the model on the training data and validate it on the validation data. Once we are satisfied with the model’s performance on the validation set, we can use it for making predictions on the test data.
### 2: Defining the model’s architecture  
We have to define how our model will look and that requires answering questions 
•	How many layers do we want?
•	What should be the activation function for each layer?
•	How many hidden units should each layer have?
These are essentially the hyperparameters of the model which play a MASSIVE part in deciding how good the predictions will be.
### 3: Training the model  
For training the model, we require:
•	Training images and their corresponding true labels
•	Validation images and their corresponding true labels 
We also define the number of epochs in this step  
### 4: Estimating the model’s performance
Finally, we load the test data (images). We then predict the classes for these images using the trained model  
### Steps to Build the Model (in google colab)
1.	Setting up Google Colab
2.	Importing Libraries
3.	Loading and Preprocessing Data 
4.	Creating a validation set
5.	Defining the model structure 
6.	Training the model 
7.	Making predictions 
#### 1: Setting up Google Colab
we’re importing our data from a Google Drive link, and we need to add a code in our Google Colab in a new Python 3 notebook and write the following code:  
```
!pip install PyDrive
```
This will install PyDrive. Now we import required libraries:
```
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth

```
Next, we will create a drive variable to access Google Drive:
```
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

```
To download the dataset, we will use the ID of the file uploaded on Google Drive:
```
download = drive.CreateFile({'id': '1BZOv422XJvxFUnGh-0xVeSvgFgqVY45q'})
```
Replace the ‘id’ in the above code with the ID of your file. Now we will download this file and unzip it:
```
download.GetContentFile('train-lable.zip')
!unzip train-lable.zip
```
#### 2 : Import the libraries for our model 
```
import keras
import numpy as np
import pandas as pd
import pytorch
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import matplotlib.pyplot as plt
```
#### 3: loading the .csv file and data.
```
train = pd.read_csv('train.csv')
```
Next, we will read all the training images, store them in a list, and convert that list into a numpy array.
```
# for grayscale images, grayscale=True, if you have RGB images, you should set grayscale=False.
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
```
As it is a multi-class classification problem (? classes),...
```
 X = np.array(train_image)
 y = train['label'].values
```
#### 4: Creating a validation set from the training data.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
```
 
#### 5: Define the model structure.





