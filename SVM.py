#!/usr/bin/python
# -*- coding: utf-8 -*-

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

#DataPath of your CICDDOS CSV files.

DataPath = '/home/abdullah/Downloads/CSV-03-11/03-11'
#Get List of files in this directory by names.
FilesList = os.listdir(DataPath)

count = 0
#DataPath of a folder to export images into it.
dstpath = '/home/abdullah/Desktop/IM/Training'
dstpath2 = dstpath + 'Benign_'

cicids_data = []
for FileName in FilesList:
  if FileName.endswith(".csv"):
    print(FileName)
    cicids_data.append(pd.read_csv(DataPath +'/'+FileName, low_memory=Fals))
#print(cicids_data)    
cicids_data = pd.concat(cicids_data)
cicids_data = cicids_data.rename(columns={'Label': 'Label'})
dataframe=cicids_data.copy()

#print(dataframe)
print(dataframe.head(10))
print('sucess')

dataframe.to_csv('data.csv')
#reading data from data.csv, usecols for reading only specified features
df = pd.read_csv("data.csv", low_memory=False)
df.drop(labels=['Unnamed: 0', 'Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port','SimillarHTTP', ' Timestamp'], axis=1, errors='ignore', inplace=True)
#Replacing the infinity values with NaN.


df = df.replace([np.inf, -np.inf], np.nan)
#Dropping NaN values.
df.dropna(inplace=True)#axis : {0 or ‘index’, 1 or ‘columns’}, default 0

#df = df.rename(columns={' Label': 'Label'})
df.loc[df['label'] != 'BENIGN', 'label'] = 1
df.loc[df['label'] == 'BENIGN', 'label'] = 0

print ("number of colummns %d" %(len(df.columns.values)))
print ("number of rows %d" %(len(df.index.values)))
#print availbe classes after filtering
print(df['label'].count())
#using shuffle for training data, it is recommended to avoid having the normal traffic or attack traffic in a sequence
from sklearn.utils import shuffle
df = shuffle(df)
df = shuffle(df)
df = shuffle(df)
df = shuffle(df)
df = shuffle(df)

df = df.reset_index()
del df['index']

print(df.head())
X = df.drop('label', axis=1)
y = df['label']
#X = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df), columns=df.columns, index=df.index)
#print(X)
X = np.array(X).astype(np.float32)
y=y.astype('int')
#print(y)
#Dividing the attack traffic 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
svclassifier = SVC(C=0.001, kernel='linear')
import time
start = time.time()
svclassifier.fit(X_train, y_train)
stop = time.time()
print("The time of the run:", stop - start)

y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print ("Accuracy : ", accuracy_score(y_test, y_pred))
