#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:19:08 2021

@author: sam
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Case 1: [A]<->[B]
stable
"""

A0 = 50;
B0 = 20; 
k1 = 0.045;
k2 = 0.055; 
t_max =50;

A_memory = np.zeros([1,t_max]);
B_memory = np.zeros([1,t_max]);

for i in range(0,t_max):
    if i==0:
        A_memory[0,i]=A0;
        B_memory[0,i]=B0;
    else:
         dA= -k1*A_memory[0,i-1] + k2*B_memory[0,i-1]
         dB = k1*A_memory[0,i-1] -k2*B_memory[0,i-1]
         
         A_memory[0,i] = A_memory[0,i-1]+dA;
         B_memory[0,i] = B_memory[0,i-1]+dB;
         
plt.plot(np.transpose(A_memory),'go--', linewidth=2, markersize=4)
plt.plot(np.transpose(B_memory),'ro--', linewidth=2, markersize=4)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Law of Mass action in system [A]<->[B]')
plt.show();

"""
Case 2: [A]<->[B]+[A]
unstable
"""
A0 = 50;
B0 = 20; 
k1 = 0.045;
k2 = 0.055; 
t_max =10000;

A_memory = np.zeros([1,t_max]);
B_memory = np.zeros([1,t_max]);

for i in range(0,t_max):
    if i==0:
        A_memory[0,i]=A0;
        B_memory[0,i]=B0;
    else:
         dA= -k1*A_memory[0,i-1] + k2*B_memory[0,i-1] +k2*0.01*A_memory[0,i-1]
         dB = k1*A_memory[0,i-1] -k2*B_memory[0,i-1]
         
         A_memory[0,i] = A_memory[0,i-1]+dA;
         B_memory[0,i] = B_memory[0,i-1]+dB;
         
plt.plot(np.transpose(A_memory),'go--', linewidth=2, markersize=2)
plt.plot(np.transpose(B_memory),'ro--', linewidth=2, markersize=2)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Law of Mass action in system [A]<->[B]+0.01[A]')
plt.show();

"""
Case 3: 
    [A]->[B]->[C]+[D]
    [C]->[A]
    [D]->[B]
Intermediary Points
"""

A0 = 50;
B0 = 0;
C0 = 0;
D0 = 0;

k1 = 0.7
k2 = 0.2
k3 = 0.3
k4 = 0.2
t_max = 50;

A_memory = np.zeros([t_max])
B_memory = np.zeros([t_max])
C_memory = np.zeros([t_max])
D_memory = np.zeros([t_max])

for i in range(0,t_max):
    if i ==0:
        A_memory[i]=A0;
        B_memory[i]=B0;
        C_memory[i]=C0;
        D_memory[i]=D0;
    else:
        dA = -k1*A_memory[i-1] + k3*C_memory[i-1]
        dB = k1*A_memory[i-1] + k4*D_memory[i-1]-k2*B_memory[i-1]
        dC = 0.5*k2*B_memory[i-1]-k3*C_memory[i-1]
        dD = 0.5*k2*B_memory[i-1]-k4*D_memory[i-1]
        
        A_memory[i] = A_memory[i-1]+dA;
        B_memory[i] = B_memory[i-1] + dB;
        C_memory[i] = C_memory[i-1] +dC;
        D_memory[i] = D_memory[i-1] +dD;
        
plt.plot(np.transpose(A_memory),'go--', linewidth=2, markersize=4)
plt.plot(np.transpose(B_memory),'ro--', linewidth=2, markersize=4)
plt.plot(np.transpose(C_memory),'bo--', linewidth=2, markersize=4)
plt.plot(np.transpose(D_memory),'yo--', linewidth=2, markersize=4)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Law of Mass action in the 4D system')
plt.show();

"""
Case 4: Encoding extinction 
        [A]->[B]->[C]+[D]
        [C]->[A]
        [D]->[B]
        if [A] < 3 then dA=0
"""

A0 = 50;
B0 = 0;
C0 = 0;
D0 = 0;

k1 = 0.7
k2 = 0.2
k3 = 0.3
k4 = 0.2
t_max = 50;

A_memory = np.zeros([t_max])
B_memory = np.zeros([t_max])
C_memory = np.zeros([t_max])
D_memory = np.zeros([t_max])

for i in range(0,t_max):
    if i ==0:
        A_memory[i]=A0;
        B_memory[i]=B0;
        C_memory[i]=C0;
        D_memory[i]=D0;
    else:
        if A_memory[i-1]<3:
            dA=0; 
            A_memory[i]=0
        else:
            dA = -k1*A_memory[i-1] + k3*C_memory[i-1]
            A_memory[i] = A_memory[i-1]+dA;
        dB = k1*A_memory[i-1] + k4*D_memory[i-1]-k2*B_memory[i-1]
        dC = 0.5*k2*B_memory[i-1]-k3*C_memory[i-1]
        dD = 0.5*k2*B_memory[i-1]-k4*D_memory[i-1]
        
        B_memory[i] = B_memory[i-1] + dB;
        C_memory[i] = C_memory[i-1] +dC;
        D_memory[i] = D_memory[i-1] +dD;
        
plt.plot(np.transpose(A_memory),'go--', linewidth=2, markersize=4)
plt.plot(np.transpose(B_memory),'ro--', linewidth=2, markersize=4)
plt.plot(np.transpose(C_memory),'bo--', linewidth=2, markersize=4)
plt.plot(np.transpose(D_memory),'yo--', linewidth=2, markersize=4)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Law of Mass action in 4D system with extinction ')
plt.show();
