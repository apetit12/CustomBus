#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 07:27:51 2019

@author: antoinepetit

Data was retrieved from:
https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p
"""

import utm
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from dbscan_ import DBSCAN

df = pd.read_csv("xaa.csv")

df.loc[:,'Trip Start Timestamp'] = df['Trip Start Timestamp'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y %I:%M:%S %p'))
df.loc[:,'Trip End Timestamp'] = df['Trip End Timestamp'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y %I:%M:%S %p'))
df = df.drop('Trip ID',axis=1)

df = df.dropna(subset=['Pickup Centroid Latitude', 'Dropoff Centroid Latitude'])
###############################################################################
# Select trips from Nov 01, 2018
###############################################################################
df['Trip Start Date'] = df['Trip Start Timestamp'].apply(lambda x: x.date())

df_sample = df[df['Trip Start Date'] == datetime(2018,11,1).date()]
df_sample = df_sample[['Trip Start Timestamp','Trip End Timestamp','Pickup Centroid Latitude',
                       'Pickup Centroid Longitude','Dropoff Centroid Latitude','Dropoff Centroid Longitude']]
#df_sample['Trip Start Time'] = df_sample['Trip Start Timestamp'].apply(lambda x: 60*x.time().hour+x.time().minute)

###############################################################################
# Look at the temporal distribution of the trips
###############################################################################
df_time = df_sample[['Trip Start Timestamp','Trip End Timestamp']]
df_time['Stime'] = df_sample['Trip Start Timestamp'].apply(lambda x: str(x.time()))
df_time['Etime'] = df_sample['Trip End Timestamp'].apply(lambda x: str(x.time()))
df_time = df_time[['Stime','Etime']]

grouped_time = df_time.groupby('Stime').agg('count')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
grouped_time.plot(y='Etime',ax=ax1,legend=False)
plt.xlabel('Time of day')
plt.title('Number of riders per time of day on 11/01/18')

###############################################################################
# Look at the spatial distribution of the trips
###############################################################################
df_loc = df_sample[['Pickup Centroid Latitude','Pickup Centroid Longitude','Dropoff Centroid Latitude','Dropoff Centroid Longitude']]

#n_sample = 10000
XX_o = df_loc['Pickup Centroid Longitude'].values
YY_o = df_loc['Pickup Centroid Latitude'].values
XX_d = df_loc['Dropoff Centroid Longitude'].values
YY_d = df_loc['Dropoff Centroid Latitude'].values
#index = np.random.choice(XX_p.shape[0], n_sample, replace=False)  
#xrandom_p = XX_p[index]
#yrandom_p = YY_p[index]

#XX_d = df_loc['Dropoff Centroid Longitude'].values
#YY_d = df_loc['Dropoff Centroid Latitude'].values
#index = np.random.choice(XX_d.shape[0], n_sample, replace=False)  
#xrandom_d = XX_d[index]
#yrandom_d = YY_d[index]
plt.figure(2)
plt.subplot(121)
plt.scatter(XX_o,YY_o,marker='.',color='r')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Trip origins on 11/01/18')
plt.subplot(122)
plt.scatter(XX_d,YY_d,marker='.',color='b')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Trip destinations on 11/01/18')

###############################################################################
# Perform DBSCAN on trip logs
###############################################################################
df_sample['raw_coords'] = df_sample[['Pickup Centroid Latitude','Pickup Centroid Longitude','Dropoff Centroid Latitude','Dropoff Centroid Longitude']]\
                .apply(lambda x: [utm.from_latlon(x[0], x[1]), utm.from_latlon(x[2], x[3])], axis=1)

non_outlier = df_sample['raw_coords'].map(lambda x: x[0][-1] + x[1][-1]) == 'TT' # To make sure all coords are in zone T.

df_sample['coords'] = df_sample['raw_coords'].apply(lambda x: list(x[0][:2]) + list(x[1][:2]))

all_coords = np.stack(df_sample['coords'])

df_sample['pickup_datetime'] = df_sample['Trip Start Timestamp'].apply(lambda x: np.datetime64(x))
df_sample['dropoff_datetime'] = df_sample['Trip End Timestamp'].apply(lambda x: np.datetime64(x))

all_timestamps = df_sample[['pickup_datetime','dropoff_datetime']].values

db = DBSCAN(eps_d=1000, eps_t=20/1.66667e-11, min_samples=4, metric_d='l2', metric_t='l1').fit(all_coords, all_timestamps)

################################################################################
## Plot some of the clusters
################################################################################
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
label_count = list(zip(*np.unique(db.labels_, return_counts=True)))
label_count.sort(key=lambda x: x[1],reverse=True)
#print(label_count[:10])

N_plot = 5
df_sample['label'] = db.labels_
colorst = plt.cm.rainbow(np.linspace(0, 0.9,N_plot))

for ii in range(1,N_plot):
    fig1 = plt.figure(3+ii)
    ll = label_count[ii]
    temp = df_sample[df_sample['label']==ll[0]]
    XX_1 = temp['Pickup Centroid Longitude'].values
    YY_1 = temp['Pickup Centroid Latitude'].values
    plt.subplot(121)
    plt.scatter(XX_1,YY_1,marker='.',color=colorst[ii-1])
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    ttt = temp.head(1)['Trip Start Timestamp'].values[0]
    plt.title('Origins cluster #'+str(ii) + ' around ' + str(ttt)[11:19])
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')

    XX_2 = temp['Dropoff Centroid Longitude'].values
    YY_2 = temp['Dropoff Centroid Latitude'].values
    plt.subplot(122)
    plt.scatter(XX_2,YY_2,marker='.',color=colorst[ii-1])
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.title('Destinations cluster #'+str(ii))
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')

plt.figure(100)
xx = [x[1] for x in label_count[2:]]
plt.hist(xx, normed=False, bins=len(label_count)-2)
plt.ylabel('Number of clusters')
plt.xlabel('Number of points in cluster')
plt.title('Cluster size distribution')