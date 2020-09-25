# Simple-Image-Classification-Model
Build your First Simple Image Classification Model 
### What is Image Classification?  
There are potentially n number of categories in which a given image can be classified. Manually checking and classifying images is a very boring process.  
The task becomes near impossible when we’re faced with a massive number of images, How useful would it be if we could automate this entire process and quickly label images per their corresponding class?
### Setting up the Structure of our Image Data  
Our data needs to be in a particular format in order to solve an image classification problem.
You should have 2 folders, one for the train set and the other for the test set. In the training set, you will have a .csv file and an image folder:
•	The .csv file contains the names of all the training images and their corresponding true labels
•	The image folder has all the training images.
The .csv file in our test set is different from the one present in the training set. This test set .csv file contains the names of all the test images, but they do not have any corresponding labels. Our model will be trained on the images present in the training set and the label predictions will happen on the testing set images
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
### Steps to Build our Model
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



