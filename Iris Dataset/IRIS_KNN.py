#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:50:21 2019

@author: sam
"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data, classifier = load_iris(return_X_y=True)
# for ease of visualisation, i reduce the dimenions to 2D
data = np.delete(data,0,1)
data = np.delete(data,0,1)


neigh = KNeighborsClassifier(n_neighbors=1)
holdout = 0.5; 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, classifier, test_size=holdout)
neigh.fit(Xtrain,Ytrain)
Ypred = neigh.predict(Xtest)
score = (30-sum(abs(Ytest-Ypred)))/30


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data,classifier)
# test the seperatrix with synthetic data
synth = np.zeros([816,2])
synth_classifier = np.zeros([816,1])
t0=0;

for k in range(1,69,2):
    for l in range(1,25,1):
        synth[t0,:] = [0.1*k,0.1*l]
        synth_classifier[t0] = neigh.predict([synth[t0,:]])
        
        t0=t0+1;
    
# Generate the plot
species = ['Setosa','Versicolor','Virginica']
for i in range(0,3):
    plt.scatter(synth[np.where(synth_classifier==i),0],synth[np.where(synth_classifier==i),1], label=species[i])
    
plt.legend()
plt.xlabel('Petal Length [cm]')
plt.ylabel('Petal Width [cm]')
plt.title('5-NN of the Iris dataset')
plt.show()




