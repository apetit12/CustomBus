#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 07:27:51 2019

@author: antoinepetit
"""

import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

df = pd.read_csv("xaa.csv")

df['Trip Start Timestamp'] = df['Trip Start Timestamp'].apply(lambda x: datetime.strptime(x,'%m/%d/%y %H:%M'))
df['Trip End Timestamp'] = df['Trip End Timestamp'].apply(lambda x: datetime.strptime(x,'%m/%d/%y %H:%M'))
df = df.drop('Trip ID',axis=1)

# Look at the temporal distribution of the trips
df_time = df[['Trip Start Timestamp','Trip End Timestamp']]
df_time['Stime'] = df['Trip Start Timestamp'].apply(lambda x: str(x.time()))
df_time['Etime'] = df['Trip End Timestamp'].apply(lambda x: str(x.time()))
df_time = df_time[['Stime','Etime']]

grouped_time = df_time.groupby('Stime').agg('count')
grouped_time.plot(y='Etime',legend=False)
plt.xlabel('Time of day')
plt.title('Number of riders')

'''
This plot can be a first step towards planning the necessary fleet size of 
customized buses.
'''

# Look at the spatial distribution of the trips
df_loc = df[['Pickup Centroid Latitude','Pickup Centroid Longitude','Dropoff Centroid Latitude','Dropoff Centroid Longitude']]

n_sample = 10000
XX_p = df_loc['Pickup Centroid Longitude'].values
YY_p = df_loc['Pickup Centroid Latitude'].values
index = np.random.choice(XX_p.shape[0], n_sample, replace=False)  
xrandom_p = XX_p[index]
yrandom_p = YY_p[index]

#XX_d = df_loc['Dropoff Centroid Longitude'].values
#YY_d = df_loc['Dropoff Centroid Latitude'].values
#index = np.random.choice(XX_d.shape[0], n_sample, replace=False)  
#xrandom_d = XX_d[index]
#yrandom_d = YY_d[index]

plt.scatter(xrandom_p,yrandom_p,marker='^',color='b')
#plt.scatter(xrandom_d,yrandom_d,marker='.',color='r')
plt.xlabel('Latitude')
plt.ylabel('Longitude')