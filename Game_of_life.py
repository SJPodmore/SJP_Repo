#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:49:13 2021

@author: sam

This code was written to emulate the game of life. 
"""
import numpy as np
 
 """
 Defines the initial conditions for the game 
 """
Landscape = np.zeros([5,5])
Landscape[1,2]=1;
Landscape[2,2]=1;
Landscape[3,2]=1;


def Iterate(Landscape):
    Landscape_new = np.zeros([np.size(Landscape,0),np.size(Landscape,1)]);
    
    for i in range(0,np.size(Landscape,0)):
        for j in range(0,np.size(Landscape,1)):
            
            #Case 1
            if Landscape[i,j] == 1:
                Count_ = Count(Landscape,i,j)
                if Count_==2 or Count_==3:
                    Landscape_new[i,j]=1
            #Case 2
            if Landscape[i,j]==0:
                Count_ = Count(Landscape,i,j)
                if Count_==3:
                    Landscape_new[i,j]=1;
            
            
    return Landscape_new; 

def Count(Landscape, i,j):
    sum_count = 0;
    for m in range(-1,2,1):
        for n in range(-1,2,1):
            if n!=0 or m!=0: 
                if 0<=i-m<np.size(Landscape,0) and 0<=j-n<np.size(Landscape,1):
                    sum_count += Landscape[i-m,j-n];
    return sum_count;


def display(Landscape):
    
    return 0;

for iterate in range(0,5):
    
    Landscape = Iterate(Landscape)
    print(Landscape)
    print(" ")
