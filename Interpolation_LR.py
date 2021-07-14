#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sam Podmore
"""

import numpy as np

def Gen_test_1(X_input):
    Y_output = 2*X_input;
    return Y_output;

def Gen_test_2(X_input):
    
    Y_output = np.inner([2,1],X_input);
    return Y_output;

def Gen_test_3(X_input):
    Y_output = X_input**2
    return Y_output;

def Linear_regression(y,x):
    if np.shape(np.shape(x)) == (1,):
        L_LR = 1/np.inner(x,x)
        R_LR = np.inner(x,y);
    else:
        L_LR = np.linalg.pinv(np.inner(np.transpose(x),np.transpose(x)));
        R_LR = np.inner(np.transpose(x),y);
    M = np.inner(L_LR,R_LR); 
    return M;

def RMSE(Y_test,Y_approx):
    Error = np.sqrt(np.mean((Y_test - Y_approx)**2))
    return Error;
"""
Test 1: y =2x
Training set: x={-100:1:100}
Test set: x={150:1:200}
"""

X_train = np.linspace(-100,100,101)
Y_train = Gen_test_1(X_train)
M1 = Linear_regression(Y_train, X_train)

X_test = np.linspace(150,200,51)
Y_test = Gen_test_1(X_test);
Y_approx = np.inner(M1,np.transpose(X_test));
Test_1_error = RMSE(Y_test,Y_approx)

"""
Test 2: y =mx | m = [2/3,1/3]
Training set: x1= {-100:1:100}, x2={-100:1:100}
Test set: x1= {150:1:200},x2={150:1:200}
"""
X_train=np.append(np.reshape(np.linspace(-100,100,101),(101,1)),np.reshape(np.linspace(-100,100,101),(101,1)),axis=1)
Y_train = Gen_test_2(X_train);
M2 = Linear_regression(Y_train, X_train)

X_test=np.append(np.reshape(np.linspace(150,200,51),(51,1)),np.reshape(np.linspace(150,200,51),(51,1)),axis=1)
Y_test = Gen_test_2(X_test);
Y_approx = np.inner(M2,X_test);
Test_2_error = RMSE(Y_test,Y_approx)

"""
Test 2.2: y =mx | m = [2/3,1/3]
Training set: x1= {-100:1:100}, x2={-80:1:120}
Test set: x1= {150:1:200},x2={150:1:200}

"""
X_train=np.append(np.reshape(np.linspace(-80,120,101),(101,1)),np.reshape(np.linspace(-100,100,101),(101,1)),axis=1)
Y_train = Gen_test_2(X_train);
M2_2 = Linear_regression(Y_train, X_train)

X_test=np.append(np.reshape(np.linspace(150,200,51),(51,1)),np.reshape(np.linspace(150,200,51),(51,1)),axis=1)
Y_test = Gen_test_2(X_test);
Y_approx = np.inner(M2_2,X_test);
Test_2_2_error = RMSE(Y_test,Y_approx)

"""
Test 3: y=x**2
Training set: x={-100:1:100}
Test set: x={-100:1:100}
"""

X_train = np.linspace(-100,100,201)
Y_train = Gen_test_3(X_train)
M3 = Linear_regression(Y_train, X_train)

X_test = np.linspace(-100,100,201)
Y_test = Gen_test_3(X_test);
Y_approx = np.inner(M3,np.transpose(X_test));
Test_3_error = RMSE(Y_test,Y_approx)


"""
Test 4: y=x**2
Training set: x={-80:1:120}
Test set: x={-80:1:120}
"""

X_train = np.linspace(-80,120,201)
Y_train = Gen_test_3(X_train)
M4 = Linear_regression(Y_train, X_train)

X_test = np.linspace(-80,120,201)
Y_test = Gen_test_3(X_test);
Y_approx = np.inner(M4,np.transpose(X_test));
Test_4_error = RMSE(Y_test,Y_approx)



