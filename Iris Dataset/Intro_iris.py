#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 01:23:29 2019

@author: sam
"""

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import statistics

data, classifier = load_iris(return_X_y=True)


statistics.median(data[np.where(classifier==2)][:,3])
min(data[np.where(classifier==2)][:,3])
max(data[np.where(classifier==2)][:,3])


species = ['Setosa','Versicolor','Virginica']
for i in range(0,3):
    plt.scatter(data[np.where(classifier==i),2],data[np.where(classifier==i),3], label=species[i])
    
plt.legend()
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.title('Petal length v Petal width within the Iris dataset')
plt.show()

