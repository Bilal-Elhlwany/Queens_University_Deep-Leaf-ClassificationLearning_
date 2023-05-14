#!/usr/bin/env python
# coding: utf-8

# # CISC 867: Project 1

# ## Part I: Data Preparation
# * In this project, you will use the Leaf Classification dataset using a neural network architecture.
# * You can use keras/tensorflow or write your own routines.

# ### 1) First, download the data file, load it, and
# 1. Describe the data
# 2. Clean the data
# 3. Check the data for missing values or duplicates and carry out proper correction methods
# 4. Visualize the data using proper visualization methods.
# 5. Draw some of the images
# 6. Carry out required correlation analysis
# ### 2) Encode the labels
# ### 3) Decide if you need to standardize the data, by computing the mean and standard deviation for each feature dimension using the training set only, then subtracting the mean and dividing by the stdev for each feature and each sample.
# ### 4) Divide the data into a training and test set using approximately 80% for training.

# ## Import the needed Libraries which we will use in our project 

# In[1]:


pip install simple-colors


# In[2]:


import datetime #datetime in Python is the combination between dates and times.
import numpy as np #NumPy is the fundamental package for scientific computing in Python. 
import pandas as pd #pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool
import seaborn as sns #Seaborn is a Python data visualization library based on matplotlib.
import tensorflow as tf #TensorFlow is an open-source end-to-end framework for building Machine Learning apps.
from simple_colors import * # to print colored , bolded and underlined text
from tensorflow import keras#Keras is a high-level, deep learning API developed by Google for implementing neural networks.
import matplotlib.pyplot as plt #Is a comprehensive library for creating static, animated, and interactive visualizations.
from keras.regularizers import l2 
from keras.models import Sequential
# sklearn is an open source data analysis library, and the gold standard for Machine Learning (ML) in the Python ecosystem.
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import Input , Dense , Flatten ,Dropout
from tensorflow.keras.optimizers import  SGD, Adam, Adagrad, Adadelta, Adamax
sns.set()  #Set aesthetic parameters in one step.


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# ### Upload the data to Leaf Classification folder in colab then load the train csv file 

# In[4]:


df = pd.read_csv("/content/drive/MyDrive/Leaf-Classification/train.csv") #using the pandas to read cs file 
df


# ### Check the columns types, there names and the chech there in nulls or not

# In[5]:


df.info() #This method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.


# ### But we can't show the information since train csv file has many number of columns.

# In[6]:


# So we will pass the following parameters 
#verbose:Whether to print the full summary.
#memory_usagebool:Specifies whether total memory usage of the DataFrame elements (including the index) should be displayed.
#show_counts:Whether to show the non-null counts.
df.info(verbose=True,max_cols=34,memory_usage=True,show_counts=True) 


# ### Describe the data

# In[7]:


df.describe()


# ### Check the data for missing values or duplicates and carry out proper correction methods

# In[8]:


#Check if there is any null at any column in df using isnull 
print(df.isnull().sum().any())


# In[9]:


df.duplicated().sum()


# * So, there is no null values or duplicated values too.

# ### Visualize the data using proper visualization methods.

# In[10]:


for i in df.columns: #Plot each column in the data 
    plt.figure()
# Adding Title to the Plot
    plt.title("Scatter Plot")
 
# Setting the X and Y labels
    plt.xlabel(i)
    plt.ylabel('Distribution of the data')
    plt.hist(df[i])

#df.hist(figsize=(25,25)) we can use this without using for loop 


# In[11]:


df.hist(figsize=(50,50));


# In[12]:


"""A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in a dataset,
analogous to a histogram.KDE represents the data using a continuous probability density curve in one or more dimensions."""
df.plot.kde()


# * As we see from the plot, the min value of most of columns is almost 0 and the max value is almost 1,so we don't need to normalize our data
# * Although its known that kde is continuous probability density curve, it looks like verticle line means that the data in the same range 

# ### Draw some of the images

# In[22]:


# imports
import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.patches as mpatches  # to draw a circle at the mean contour
from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality

# matplotlib setup
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams
rcParams['figure.figsize'] = (6, 6)      # setting default size of plots


# reading an image file using matplotlib into a numpy array
# choosed ones: 9,11, 19, 21, 29, 48, 167, 192, 218
list_images = ['9','11', '19', '21', '29', '48', '167', '192', '218']
for i in list_images:
    img = mpimg.imread('/content/drive/MyDrive/Leaf-Classification/Images/'+i+'.jpg')

    # using image processing module of scipy to find the center of the leaf
    cy, cx = ndi.center_of_mass(img)

    plt.imshow(img, cmap='tab20c')  # show me the leaf
    plt.scatter(cx, cy)           # show me its center
    plt.show()


# # Carry out required correlation analysis

# In[12]:


""" A correlation heatmap is a graphical representation of a correlation matrix 
representing the correlation between different variables.
The value of correlation can take any value from -1 to 1.
Correlation between two random variables or bivariate data does not necessarily imply a causal relationship.
So i will find correlation using heat map"""
Corr_Mat = df.corr()
Corr_Mat.style.background_gradient(cmap="BrBG")


# ## 2) Encode the labels (species column)

# * The primary need for using one-hot encoding technique is to transform or convert the categorical features into numerical features such that machine learning libraries can use the values to train the model. Although, many machine learning library internally converts them, but it is recommended to convert these categorical features explicitly into numerical features (dummy features)

# In[13]:


df.species.value_counts()


# In[14]:


from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing species column (label encoded values of bridge_types)
Y = pd.DataFrame(enc.fit_transform(df[['species']]).toarray())


# ### 3) Decide if you need to standardize the data, by computing the mean and standard deviation for each feature dimension using the training set only, then subtracting the mean and dividing by the stdev for each feature and each sample.

# * As we see from the plot, the min value of most of columns is almost 0 and the max value is almost 1,so we don't need to normalize our data
# * Although its known that kde is continuous probability density curve, it looks like verticle line means that the data in the same range

# ### 4) Divide the data into a training and test set using approximately 80% for training.

# In[15]:


X = df.iloc[:, :]
X.drop(['species', 'id'], axis = 1, inplace = True) # drop the columns of species
X.head()


# In[16]:


# Divide the data into train and test by 80% for training
xTrain, xTest, yTr, yTest = train_test_split(X, Y, test_size = 0.2, random_state = 42)

print(xTrain.shape)
print(xTest.shape)
print(yTr.shape)
print(yTest.shape)


# In[17]:


# Divide the training set itself into train set and validation set which is used to tune hyperparameters 
xTrain, X_val, yTrain, y_val = train_test_split(xTrain, yTr, test_size=0.15, random_state=42)

print(xTrain.shape)
print(xTest.shape)
print(X_val.shape)
print(y_val.shape)


# #### Part II
# ##### Training a neural network In this project, you need to implement a 3-layer MLP model (one input layer, one hidden layer with tanh activation and one output layer) which will be used to classify the data in Part I. You can use the built-in modules in keras to build your model.

# * You also need to write the training function (training), and should explore the following hyperparameter settings:
#     * Batch size: Number of examples per training iteration.
#     * Hidden size: Try using different number of hidden nodes in your model and compare the performances.
#     * Dropout is an effective strategy to defend against overfitting. Adding a dropout layer after the hidden layer, and try using different dropout rate to compare the performances.
#     * Optimizer: Try using different optimizers such as SGD, Adam, RMSProp.
#     * Regularization (weight decay): L2 regularization can be specified by setting the weight_decay parameter in optimizer. Try using different regularization factor and check the performance.
#     * Learning rate, Learning rate scheduler: Learning rate is key hyperparameter in model training, and you can gradually decreasing the learning rate to further improve your model. Try using different learning rate and different learning rate scheduler to compare the performance.

# * To get full credit, you should explore at least 4 different type of hyperparameters (from listed above), and choose at least 3 different values for each hyperparameters. For simplicity, you could analyze one hyperparameter at a time (i.e. fixing all others to some reasonable value), rather than perfoming grid search. If you use TensorBoard to monitor your training, you can directly attach the screenshots of the training curves (accuracy) in your report. To evaluate the performance of trained model, you also need to write a function (evaluation) which loads the trained model and evaluate its performance on train/test set.

# ### We will build our deep learning model using 3 function:
# * 1- Deep learning architecture(3-layer MLP model (one input layer, one hidden layer with tanh activation and one output layer)
# * 2- One function for virtualization of accuracy and val_accuracy
# * 3- One function for virtualization of loss and val_loss

# * 1- Deep learning architecture(3-layer MLP model (one input layer, one hidden layer with tanh activation and one output layer)
# * This function has multiple hyperparameters which is Batch size,Hidden size,Optimizer:, Dropout rate, Regularization and Learning rate   
# * For any function there are default values for the parameters which will be passed 

# In[18]:


"""
1- Define Keras Model
Model in Keras always defines as a sequence of layers.
It means that we initialize the sequence model and add the layers one after the other 
which is executed as the sequence of the list.
2- Compile The Keras Model
It uses the backend numerical libraries such as TensorFlow or Theano.
Whatever backend you are using automatically chooses the best way to represent the network on your hardware as CPU, GPU, or TPU.
3- Start Training (Fit the Model)
We need to define a number of epochs and batch size over which training occurs.
Epoch – one single pass through all the rows in the training dataset
Batch size – number of samples considered by the model before updating the weights.
4- Evaluate the Model
After training the model let’s know the performance of a neural network. Model is always evaluated on a test set.
5- Making Predictions
predict the output of new data by simply using predict method.
"""
def MLP_Model(Optimizer= Adam , Batch_size = 32,  Hidden_size = 512, Dropout_rate = 0.5):
    
    model = Sequential()
  # In layer_1 our activation function is 'tanh' with default 512 neurons and kernel_initializer 'glorot_uniform'
    model.add(Dense(Hidden_size, activation='tanh', input_shape=(X.shape[1],), kernel_initializer = 'glorot_uniform', bias_initializer='zeros', name = 'Layer_1'))
  #dropout some nerouns to avoid overfitting
    model.add(Dropout(Dropout_rate))
  #output layer with softmax activation function and has 99 nodes for output shape
    model.add(Dense(99 , activation='softmax', name = 'Output'))
  #Apply one of the optimizer and choose the learning rate inside each one 
    if Optimizer == 'Adam':
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
        Optimizer = Adam(learning_rate = lr_schedule)
    elif Optimizer == 'RMSprop':
        Optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01,rho=0.9,momentum=0.0,epsilon=1e-07,centered=False, name="RMSprop")
    elif Optimizer == 'SGD':
        Optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False, name="SGD")
    #compile the model with sparse_categorical_crossentropy loss function and accuracy metrics 
    model.compile(optimizer = Optimizer ,loss='CategoricalCrossentropy' , metrics=['accuracy'])

  #fit the model with 100 epoch
    history = model.fit(xTrain , yTrain , epochs=100 , batch_size=Batch_size , validation_data=(X_val, y_val))

  #return the training model and the history
    return model, history

def Virtualization_Acc(model_name, history):
    model_history = pd.DataFrame(history.history)
    model_history['epoch'] = history.epoch
    fig, ax = plt.subplots(figsize=(14,8))
    num_epochs = model_history.shape[0]
    ax.plot(np.arange(0, num_epochs), model_history["accuracy"], label="Training accuracy", lw=3, color='#f4b400')
    ax.plot(np.arange(0, num_epochs), model_history["val_accuracy"], label="Validation accuracy", lw=3, color='#0f9d58')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def Virtualization_Loss(model_name, history):
    model_history = pd.DataFrame(history.history)
    model_history['epoch'] = history.epoch
    fig, ax = plt.subplots(figsize=(14,8))
    num_epochs = model_history.shape[0]
    ax.plot(np.arange(0, num_epochs), model_history["loss"], label="Training loss", lw=3, color='#f4b400')
    ax.plot(np.arange(0, num_epochs), model_history["val_loss"], label="Validation loss", lw=3, color='#0f9d58')
    ax.legend()
    plt.tight_layout()
    plt.show()


# In[21]:


# Build and train the model with different hyperparameters
iter = 0
for batch_size in [16, 32, 64]:
    for hidden_size in [64,128, 256]:
        for dropout_rate in [0.2, 0.4, 0.6]:
            for optimizer_name in ['SGD', 'Adam', 'RMSprop']:
                print(blue(f'The batch_size is: {batch_size}, The hidden_size is: {hidden_size}, The dropout_rate is: {dropout_rate}, The optimizer_name is : {optimizer_name}',['bold', 'underlined']) )
                model_1, history_1 = MLP_Model(Optimizer= optimizer_name , Batch_size = batch_size,  Hidden_size = hidden_size, Dropout_rate = dropout_rate)
                model_1.summary()
                model_1.evaluate(xTest, yTest)
                #virtualization for the training accuracy and validation accuracy throw the epochs number
                Virtualization_Acc(model_1,history_1)
                #virtualization for the training loss and validation loss throw the epochs number
                Virtualization_Loss(model_1,history_1)
                print(blue('============================================================================================================',['bold']))
                iter = iter+1
                print(red(iter,['bold']))
                print('                                                                                                                            ')


# # Thank You
