#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 19:54:36 2021

@author: sam
"""
import numpy as np 
import pandas as pd
import seaborn as sns 
import contextily 
import matplotlib.pyplot as plt
import scipy as sci

dat = pd.read_csv("/Users/sam/Documents/Scripts_Backup/Scripts/PythonScripts/london.csv")


# Clean the data for roughly zone 1, London 
dat = dat.drop(dat[dat.Price==0].index)
dat = dat.drop(dat[dat.Latitude<51.45].index)
dat = dat.drop(dat[dat.Latitude>51.55].index)
dat = dat.drop(dat[dat.Longitude<-0.2].index)
dat = dat.drop(dat[dat.Longitude>0].index)


def Joint_histogram(dat):
    # Generate scatterplot + histograms of property long/lat     
    joint_axes = sns.jointplot(x=dat['Longitude'],y=dat['Latitude'],kind='scatter',s=2)
    contextily.add_basemap(joint_axes.ax_joint,crs="EPSG:4326",source=contextily.providers.CartoDB.PositronNoLabels);

def Hex_map(dat):
    #Generate 2D Hitstogram
    f, axes = plt.subplots(1, figsize=(12, 9))
    # Generate and add hexbin with 30 hexagons each
    hist = axes.hexbin(
        dat['Longitude'], 
        dat['Latitude'],
        gridsize=30, 
        linewidths=0,
        alpha=0.4, 
        cmap='viridis_r'
        )
    # Add basemap
    contextily.add_basemap(axes,crs="EPSG:4326",source=contextily.providers.CartoDB.Positron)
    plt.colorbar(hist,shrink=0.5)
    plt.title('Hex based histogram of properties with £3-4k pcm rent cost in Zone 1')
    axes.set_axis_off()
    plt.savefig('/Users/sam/Documents/Hexmap_london_rent_3_4K.png', dpi=300)

def KDE(dat):
    #Generate kernal density estimation 
    f, axes = plt.subplots(1, figsize=(9, 9))
    # Generate and add KDE with a shading of 30 gradients 
    sns.kdeplot(
        dat['Longitude'], 
        dat['Latitude'],
        n_levels=50, 
        shade=True,
        alpha=0.1, 
        cmap='viridis_r'
        )
    # Add basemap
    contextily.add_basemap(axes,crs="EPSG:4326",source=contextily.providers.CartoDB.Positron)
    plt.title('Map of the Hackney area, London')
    plt.savefig('/Users/sam/Documents/KDE_London_Rent.png',dpi=300)

def plot_3d(dat,inte):
    fig = plt.figure()
    X_,Y_,Z_= interp_3d(dat)
    if inte==1:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_, Y_, Z_)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Price')
        plt.title('Surface map of rent price in zone 1, London')
        plt.show()
        plt.savefig('/Users/sam/Documents/surface_London_Rent.png',dpi=300)
    if inte==2:
        fig, ax = plt.subplots()
        heatmap = ax.pcolormesh(X_, Y_, Z_, cmap='magma')
        ax.axis([X_.min(), X_.max(), Y_.min(), Y_.max()])
        fig.colorbar(heatmap, ax=ax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.title('Heatmap of rent price in Hackney, London')
        plt.savefig('/Users/sam/Documents/Heatmap_Hackney_Rent.png',dpi=300)
        
def interp_3d(dat):
    #5-NN to interpolate surface  of rent price
    from sklearn.neighbors import KNeighborsRegressor
    neigh = KNeighborsRegressor(n_neighbors=5,weights='distance',p=2)   
    neigh.fit(dat[['Longitude','Latitude']].values, dat['Price'].values)

    X_synth = np.linspace(min(dat.Longitude),max(dat.Longitude),1000)
    Y_synth = np.linspace(min(dat.Latitude),max(dat.Latitude),1000)
    [X_,Y_] = np.meshgrid(X_synth,Y_synth);
    X_T = np.append(X_.reshape(-1,1),Y_.reshape(-1,1),axis=1)
    Z_ = neigh.predict(X_T)
    Z_ = Z_.reshape(X_.shape)
    
    return X_,Y_,Z_     

def Price_hist(dat):
   #Removing houses that cost greater than 5000  for cleaner results
   dat = dat.drop(dat[dat.Price>5000].index)
   plt.hist(dat['Price'].values,20,color='k')
   plt.xlabel('Cost (pcm)')
   plt.ylabel('Frequency (#)')
   plt.title('Histogram of house prices pcm, North of the Thames, London')
   plt.savefig('/Users/sam/Documents/Histogram_NotT_Hackney.png',dpi=300)

def KM(dat,n_components,inte):
    from sklearn.cluster import KMeans
    K = KMeans(n_clusters = n_components).fit(dat[['Longitude','Latitude','Price']].values);
    X_,Y_,Z_= interp_3d(dat)
    X_T = np.append(X_.reshape(-1,1),Y_.reshape(-1,1),axis=1)
    X_T = np.append(X_T,Z_.reshape(-1,1),axis=1)
    
    Labels_ = K.predict(X_T)
    Labels_= Labels_.reshape(X_.shape)
    
    if inte ==0:
        return X_,Y_,Z_;
    if inte ==1:
        fig, ax = plt.subplots()
        heatmap = ax.pcolormesh(X_, Y_, Labels_, cmap='cividis')
        ax.axis([X_.min(), X_.max(), Y_.min(), Y_.max()])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.title('Segmentation of rent price in zone 1, London by K-means')
        #plt.savefig('/Users/sam/Documents/K-means_London_Rent.png',dpi=300)
        
        return X_,Y_,Z_;
    
def Natural_Breaks(dat,Quantiles):
    
    dat['Breaks'] = pd.qcut(dat['Price'], q=Quantiles, labels=range(0,Quantiles))
    
    from sklearn.neighbors import KNeighborsRegressor
    neigh = KNeighborsRegressor(n_neighbors=5,weights='distance',p=2)   
    neigh.fit(dat[['Longitude','Latitude']].values, dat['Breaks'].values)

    Break_means = np.zeros([Quantiles])
    for i in range(0,Quantiles):
        Break_means[i] = np.mean(dat[dat.Breaks==i].Price.values)
        
    X_synth = np.linspace(min(dat.Longitude),max(dat.Longitude),1000)
    Y_synth = np.linspace(min(dat.Latitude),max(dat.Latitude),1000)
    [X_,Y_] = np.meshgrid(X_synth,Y_synth);
    X_T = np.append(X_.reshape(-1,1),Y_.reshape(-1,1),axis=1)
    Z_ = neigh.predict(X_T)
    Z_ = Z_.reshape(X_.shape)
    
    fig, ax = plt.subplots()
    heatmap = ax.pcolormesh(X_, Y_, Z_, cmap='cividis')
    ax.axis([X_.min(), X_.max(), Y_.min(), Y_.max()])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title('Segmentation of rent price in Hackney, London based on natural breaks')

def Hackney(dat):
    #Bethanl Green + Hackney 
    #  51.51<lat<51.53
    #  -0.075<long<-0.025
    dat = dat.drop(dat[dat.Latitude<51.53].index)
    dat = dat.drop(dat[dat.Latitude>51.55].index)
    dat = dat.drop(dat[dat.Longitude<-0.075].index)
    dat = dat.drop(dat[dat.Longitude>-0.025].index)

    return dat;

def Not_Hackney(dat):
    
    dat = dat.drop(dat[
                       (dat.Longitude>-0.075) & (dat.Longitude<-0.025) & 
                       (dat.Latitude<51.55) & (dat.Latitude>51.53)].index);
    return dat;

def North_of_Thames(dat):
    #Define to linear equations to seperate the north of the Thames. 
    dat = dat.drop(dat[(dat.Longitude<=-0.03) & (dat.Latitude<(((1/9)*dat.Longitude)+51.498))].index)
    dat = dat.drop(dat[(dat.Longitude<=-0.02) & (dat.Longitude>-0.03) & (dat.Latitude<(((-3/2)*dat.Longitude)+51.457))].index)
    dat = dat.drop(dat[(dat.Longitude>-0.02) & (dat.Latitude<(((1/9)*dat.Longitude)+51.485))].index)

    return dat;

def South_of_Thames(dat):
    # Define linear equations to seperate the south of the Thames
    dat = dat.drop(dat[(dat.Longitude<=-0.03) & (dat.Latitude>(((1/9)*dat.Longitude)+51.498))].index)
    dat = dat.drop(dat[(dat.Longitude<=-0.02) & (dat.Longitude>-0.03) & (dat.Latitude>(((-3/2)*dat.Longitude)+51.457))].index)
    dat = dat.drop(dat[(dat.Longitude>-0.02) & (dat.Latitude>(((1/9)*dat.Longitude)+51.485))].index)
    return dat;

def T3_4K(dat):
    dat = dat.drop(dat[(dat.Price<3000)].index)
    dat = dat.drop(dat[(dat.Price>=4000)].index)
    return dat;
def T2_3K(dat):
    dat = dat.drop(dat[(dat.Price<2000)].index)
    #dat = dat.drop(dat[(dat.Price>=3000)].index)
    return dat;
def T1_2K(dat):
    dat = dat.drop(dat[(dat.Price<1000)].index)
    dat = dat.drop(dat[(dat.Price>=2000)].index)
    return dat;
def T0_1K(dat):
    dat = dat.drop(dat[(dat.Price>=1000)].index)
    return dat;


print(np.shape(T2_3K(dat[dat.Longitude<-0.1]))[0])
print(np.shape(T2_3K(dat[dat.Longitude>-0.1]))[0])
#print(sci.stats.ttest_ind(North_of_Thames(dat[dat.Longitude<-0.1]).Price.values,North_of_Thames(dat[dat.Longitude>-0.1]).Price.values));
#Hex_map(T3_4K(dat))
#plot_3d(North_of_Thames(Hackney(dat)),2)
#KDE(North_of_Thames(Hackney(dat)))
#Price_hist(North_of_Thames(Not_Hackney(dat)))
#print(np.mean(Hackney(North_of_Thames(dat)).Price.values))
#print(np.mean(Not_Hackney(North_of_Thames(dat)).Price.values))
#print(sci.stats.ttest_ind(Hackney(North_of_Thames(dat)).Price.values,Not_Hackney(North_of_Thames(dat)).Price.values));

"""
print(np.mean(dat.Price.values))
print(np.mean(North_of_Thames(dat).Price.values))
print(np.mean(South_of_Thames(dat).Price.values))
print(sci.stats.ttest_ind(South_of_Thames(dat),North_of_Thames(dat)));
print(sci.stats.ttest_ind(Hackney(dat),Not_Hackney(dat)));
print(sci.stats.mannwhitneyu(Hackney(dat),Not_Hackney(dat)));
print(np.mean(dat.Price.values))
print(np.mean(Not_Hackney(dat).Price.values))
print(np.mean(Hackney(dat).Price.values))
print('The number of properties in London that cost >£5000 pcm is ' + str((1- (dat.shape[0] - sum(dat.Price.values>5000))/dat.shape[0])*100) + '% of all properties in London')
"""
