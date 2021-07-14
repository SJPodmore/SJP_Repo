#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sam Podmore
"""
from sklearn.datasets import load_iris
import numpy as np


data, classifier = load_iris(return_X_y=True)

def Linear_regression(y,x):
    #Calculates the linear relationship between x&y
    if np.shape(np.shape(x)) == (1,):
        L_LR = 1/np.inner(x,x)
        R_LR = np.inner(x,y);
    else:
        # This looks terrible but np.inner(x,y) computes x.T*y
        # Hence np.inner(x.T,x.T) == x.T.T*x.T == x.T*x
        # This is because the approach expects NxK but was given KxN
        L_LR = np.linalg.pinv(np.inner(np.transpose(x),np.transpose(x)));
        R_LR = np.inner(np.transpose(x),y);
    M = np.inner(L_LR,R_LR); 
    return M;

def RMSE(Y_test,Y_approx):
    Error = np.sqrt(np.mean((Y_test - Y_approx)**2))
    return Error;

def solve_attribute(y,m,x):
    #Solves for a single attribute missing
    initial = y; 
    for i in range(0, np.shape(x)[0]):
        if np.isnan(x[i]):
            index_ = i
        else:
            initial=- m[i]*x[i]
        
    X_est = initial/m[index_];
    return X_est;

def Generate_test_sample(y,x,m,N):
    
    Sample_tests = np.zeros([N])
    for i in range(0,N):
        
        Rand_loc = np.random.randint(0,150);
        Rand_int = np.random.randint(0,3);
        new_x = np.array(x[Rand_loc,:][:]);
        new_x[Rand_int] = np.NaN;
        
        X_est = solve_attribute(y[Rand_loc],m,new_x);
        Sample_tests[i] = (abs(x[Rand_loc,Rand_int]-X_est)/(x[Rand_loc,Rand_int]));

    return Sample_tests; 

"""
Test: solve for m,issing values in 
Solve for M in iris Setosa
Solve LOOCV
"""
Error_array = np.zeros(np.shape(classifier))
for i in range(0,np.shape(data)[0]):
    X_train = np.concatenate((data[:i],data[i+1:]))
    Y_train = np.concatenate((classifier[:i],classifier[i+1:]))
    M = Linear_regression(Y_train,X_train);
    Y_approx = np.round(np.inner(M,X_train))
    Error_array[i] = ((abs(Y_approx)-abs(Y_train))>0).sum()
    
"""
Test 2:Interpolate 250 randomly determined unknown values
"""
X_train = np.array(data);
Y_train = np.array(classifier);
M = Linear_regression(Y_train,X_train);

Ttl_Err = Generate_test_sample(Y_train,X_train,M,1000)
print("The median error over the 1000 sample interpolation is " + str(100*np.median(Ttl_Err)-100) + "% with min " + str(100*np.min(Ttl_Err)-100) + '% and max ' + str(100*np.max(Ttl_Err)-100) + "%") 