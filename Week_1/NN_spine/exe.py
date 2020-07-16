# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:51:50 2019

@author: Max
"""

import numpy as np

# Load data set and code labels as 0 = ’NO’, 1 = ’DH’, 2 = ’SL’
labels = [b'NO', b'DH', b'SL']
data = np.loadtxt('column_3C.dat', converters={6: lambda s: labels.index(s)} )

# Separate features from labels
x = data[:,0:6]
y = data[:,6]

# Divide into training and test set
training_indices = list(range(0,20)) + list(range(40,188)) + list(range(230,310))
test_indices = list(range(20,40)) + list(range(188,230))

trainx = x[training_indices,:]
trainy = y[training_indices]
testx = x[test_indices,:]
testy = y[test_indices]

def L2_distance(x,y):
    return np.sqrt(np.sum(np.square(np.subtract(x,y))))

def L1_distance(x,y):
    return np.sum(np.abs(np.subtract(x,y)))
    
def find_NN_L1(x):
    # Compute distances from x to every row in train_data
    distances = [L1_distance(x,trainx[i,]) for i in range(len(trainy))]
    # Get the index of the smallest distance
    return np.argmin(distances)

def find_NN_L2(x):
    # Compute distances from x to every row in train_data
    distances = [L2_distance(x,trainx[i,]) for i in range(len(trainy))]
    # Get the index of the smallest distance
    return np.argmin(distances)

def NN_L1_classifier(tarinx, trainy, testx):
    # Get the index of the the nearest neighbor
    class_L1 = trainy[[find_NN_L1(testx[i,]) for i in range(len(testx))]]
    # Return its class
    return class_L1

def NN_L2_classifier(tarinx, trainy, testx):
    # Get the index of the the nearest neighbor
    class_L2 = trainy[[find_NN_L2(testx[i,]) for i in range(len(testx))]]
    # Return its class
    return class_L2


