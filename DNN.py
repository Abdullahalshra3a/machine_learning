#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import os, sys
import pandas as pd
import numpy as np
#import cv2
#from tqdm import tqdm
from sklearn import preprocessing
#import splitfolders
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore') 
#DataPath of your CICDDOS CSV files.
import random
DataPath = '/home/abdullah/Downloads/CSV-01-12/01-12'
DataPath2 = '/home/abdullah/Downloads/CSV-03-11/03-11'
#Get List of files in this directory by names.
FilesList = os.listdir(DataPath)
FilesList1 = os.listdir(DataPath2)

cicids_data = []
for FileName in FilesList:
  if FileName.endswith(".csv"):
    print(FileName)
    p = 0.01  # 1% of the lines
    df = pd.read_csv(DataPath +'/'+FileName, header=0, skiprows=lambda i: i>0 and random.random() > p, nrows=100000, low_memory=False)#
    df.drop(labels=['Unnamed: 0', 'Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port','SimillarHTTP', ' Timestamp'], axis=1, errors='ignore', inplace=True)
   #Replacing the infinity values with NaN.
    df = df.replace([np.inf, -np.inf], np.nan)
    #Dropping NaN values.
    df.dropna(inplace=True)#axis : {0 or ‘index’, 1 or ‘columns’}, default 0
    cicids_data.append(df)
for FileName in FilesList1:
  if FileName.endswith(".csv"):
    print(FileName)
    p = 0.01  # 1% of the lines
    df = pd.read_csv(DataPath2 +'/'+FileName, header=0, skiprows=lambda i: i>0 and random.random() > p, nrows=100000, low_memory=False)#
    df.drop(labels=['Unnamed: 0', 'Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port','SimillarHTTP', ' Timestamp'], axis=1, errors='ignore', inplace=True)
   #Replacing the infinity values with NaN.
    df = df.replace([np.inf, -np.inf], np.nan)
    #Dropping NaN values.
    df.dropna(inplace=True)#axis : {0 or ‘index’, 1 or ‘columns’}, default 0
    cicids_data.append(df)  
 
#print(cicids_data)    
cicids_data = pd.concat(cicids_data)
cicids_data = cicids_data.rename(columns={' Label': 'label'})
dataframe=cicids_data.copy()

#print(dataframe)
print(dataframe.head(10))
print('sucess')

#X_test = sc_X.fit_transform(X_test)
dataframe.to_csv('data.csv')
#reading data from data.csv, usecols for reading only specified features
#df = pd.read_csv("data.csv", low_memory=False)
df = dataframe
df.drop(labels=['Unnamed: 0', 'Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port','SimillarHTTP', ' Timestamp'], axis=1, errors='ignore', inplace=True)
#Replacing the infinity values with NaN.

df = df.replace([np.inf, -np.inf], np.nan)
#Dropping NaN values.
df.dropna(inplace=True)#axis : {0 or ‘index’, 1 or ‘columns’}, default 0

#df = df.rename(columns={' Label': 'Label'})
df.loc[df['label'] != 'BENIGN', 'label'] = 0
df.loc[df['label'] == 'BENIGN', 'label'] = 1


print ("number of colummns %d" %(len(df.columns.values)))
print ("number of rows %d" %(len(df.index.values)))
#print availbe classes after filtering
print(df['label'].count()) 

#extracting the features and labels from the dataframe

X, y  = df.drop('label', axis=1), df.pop('label').values
X = X.astype('float32')
y = np.array(y).astype(int)



#using shuffle for training data, it is recommended to avoid having the normal traffic or attack traffic in a sequence
from sklearn.utils import shuffle
df = shuffle(df)
df = shuffle(df)
df = shuffle(df)
df = shuffle(df)
df = shuffle(df)

df = df.reset_index()
del df['index']


#Dividing the attack traffic 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
unique, counts = np.unique(y_test, return_counts=True)
print("unique, counts =", unique, counts)

# determine the number of input features
n_features = X_train.shape[1]

# define the keras model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(8, input_shape=(n_features,), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#initializing time instance to calculate the trianing time
start_time = time.time()

# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=100)

print("--- %s seconds ---" % (time.time() - start_time))

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
print('Test loss: %.3f' % loss)


predictions = model.predict(X_test)
#this step is necessary if you used to predict the labels of a 3 dimensional data
predicted = np.rint(predictions)

(unique, counts) = np.unique(predicted, return_counts=True)
print("unique, counts =", unique, counts)


# predicted = predictions
print("predicted labels are ",predicted)
print("actual labels are ",y_test)
print(predicted.dtype)
print(predicted.shape)

#calculating metrics 
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score

accuracy = accuracy_score(y_test,predicted)
print('accuracy_score is',accuracy)
precision = precision_score(y_test,predicted)
print("precision is ", precision )
recall = recall_score(y_test,predicted)
print("recall is", recall )
f1Score = f1_score(y_test,predicted)
print("f1_score is",f1Score)

confusion_matrix = confusion_matrix(y_test,predicted)
print('confusion_matrix is \n',confusion_matrix)
