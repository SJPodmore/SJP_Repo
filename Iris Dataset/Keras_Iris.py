#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:13:20 2019

@author: sam
"""

import numpy as np
np.random.seed(125)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.datasets import load_iris
from sklearn.model_selection import LeaveOneOut


data, classifier = load_iris(return_X_y=True)

data = np.delete(data,0,1)
data = np.delete(data,0,1)

score = np.zeros([len(data)])
acc = np.zeros([len(data)])
    
loocv = LeaveOneOut()
for train_index, test_index in loocv.split(data): 
    Xtrain, Xtest = data[train_index], data[test_index]
    Ytrain, Ytest = classifier[train_index], classifier[test_index]
    
#holdout = 0.2; 
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, classifier, test_size=holdout)


    model= Sequential()
    model.add(Dense(30,activation='sigmoid', input_shape=(2,)))
    model.add(Dense(30,activation='sigmoid'))
    model.add(Dense(1))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])
    model.fit(Xtrain, Ytrain,
              epochs=20)
    score[train_index],acc[train_index] = model.evaluate(Xtest, Ytest)
    
    print('Test score:', np.mean(score))
    print('Test accuracy:', np.mean (acc))
