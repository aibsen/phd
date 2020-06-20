import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *
import matplotlib.pyplot as plt
#load real light curves

snIas=None
for i in range(11):
    snIas_str = '../../data/testing/csvs/snIa_{}.csv'.format(i)
    # print(snIas_str)
    snIa = pd.read_csv(snIas_str,sep="|").dropna(axis=1)
    if snIas is None:
        snIas = pd.concat([snIas, snIa],ignore_index=True)
    else:
        snIas = snIa
snIas.columns = ["id","time","flux","fluxerr","band"]

snIbcs_str='../../data/testing/csvs/snIbc.csv'
snIbcs = pd.read_csv(snIbcs_str,sep="|").dropna(axis=1)
snIbcs.columns = ["id","time","flux","fluxerr","band"]

snIIn_str='../../data/testing/csvs/snIIn.csv'
snIIn = pd.read_csv(snIIn_str,sep="|").dropna(axis=1)
snIIn.columns = ["id","time","flux","fluxerr","band"]

snIIp_str='../../data/testing/csvs/snIIP.csv'
snIIp = pd.read_csv(snIIp_str,sep="|").dropna(axis=1)
snIIp.columns = ["id","time","flux","fluxerr","band"]

sns_list=[snIas,snIbcs,snIIn,snIIp]
sns=pd.concat(sns_list)

data = []
tags = []
ids = []


for t, sn in enumerate(sns_list):
    # print("type ", type)
    sn_tags = df_tags(sn, t)
    if len(tags)==0:
         tags=sn_tags
    else:
        tags=pd.concat([tags,sn_tags])

def construct_vectors(sns,filename,meta_file=None):
    print('\n')
    print(len(sns))

    sns_cp = sns.copy()
    group_by_id = sns_cp.groupby(['id'])['time'].agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    group_by_id["time_diff"]=group_by_id.time_max-group_by_id.time_min
    
    #ensure there is at least 30 days of obsevations
    ids_enough_obs=group_by_id[group_by_id.time_diff>30].id.values
    group_by_id = group_by_id[group_by_id.id.isin(ids_enough_obs)]
    sn_enough=sns[sns.id.isin(ids_enough_obs)]

    #ensure there is at least 3 points per band
    group_by_id_band = sn_enough.groupby(['id','band'])['time'].agg(['count']).rename(columns = lambda x : 'time_' + x).reset_index()
    #drop all lcs that have less than 3 points
    at_least_3 = group_by_id_band[group_by_id_band.time_count>=25]
    #drop also their companion band
    band_counts = at_least_3.groupby(['id']).count().reset_index()
    band_counts = band_counts[band_counts.time_count==2]
    sn_enough=sn_enough[sn_enough.id.isin(band_counts.id)]
    tags_enough = tags[tags.id.isin(band_counts.id)]

    #coonvert bands to the code's sim uses (r=0, g=1)
    #instead of lasair's (g=1, r=2)
    sn_enough.loc[sn_enough.band==2,'band'] = 0

    tags_enough["old_id"]=tags_enough['id']
    for c,i in enumerate(tags_enough.old_id.unique()):
        tags_enough.loc[tags_enough.old_id==i,'id']=c
        sn_enough.loc[sn_enough.id==i,'id'] = c
        group_by_id.loc[group_by_id.id==i,'id'] = c

    count = 0
    print(tags_enough)
    max_length = group_by_id.time_diff.max()
    max_scaled_length = int(np.ceil(128*max_length/128))
    X=np.zeros((tags_enough.shape[0],4,max_scaled_length+2))
    print(max_scaled_length)
    print(max_length)
    print(sn_enough.id.unique().shape)
    print(tags_enough.shape[0])
    print(X.shape)
    obids = tags_enough.id.unique()
    for n,objid in enumerate(obids):
        lc = sn_enough[sn_enough.id == objid]
        lc_r = lc[lc.band == 0] 
        lc_g = lc[lc.band == 1]

        lc_length = group_by_id.loc[group_by_id.id == objid, 'time_diff'].values[0]
        lc_start = group_by_id.loc[group_by_id.id == objid, 'time_min'].values[0]
        lc_stop = group_by_id.loc[group_by_id.id == objid, 'time_max'].values[0]
        scaled_lc_length=int(np.ceil(128*lc_length/128))
        lc_step = lc_length/scaled_lc_length
        print(scaled_lc_length)
        new_x = np.arange(lc_start,lc_stop+1,lc_step)
        print(new_x.shape)
        X[n,0,0:scaled_lc_length+2] = np.interp(new_x,lc_r.time, lc_r.flux)
        X[n,1,0:scaled_lc_length+2] = np.interp(new_x,lc_g.time, lc_g.flux)

        for i in range(scaled_lc_length+1):
            X[n,2,i] = np.abs(lc_r.time.values - new_x[i]).min()
            X[n,3,i] = np.abs(lc_g.time.values - new_x[i]).min()

        # plt.plot(new_x,X[n,1,0:scaled_lc_length+1],'o')
        # plt.plot(lc_g.time,lc_g.flux,'+')
        # plt.plot(new_x,X[n,3,0:scaled_lc_length+1],'-')
        # plt.show()

        # count+=1
        # if count > 10:
        # break
            
    print(X.shape)
    print(obids.shape)
    print(tags_enough.type.values.shape)

    dataset = {
    'X':X,
    'Y':tags_enough.type.values,
    'ids':obids.astype(int)
    }
    print(dataset)

    if metafile:
        tags_enough.to_csv(meta_file)
    save_vectors(dataset,filename)


metafile="real_data_tags_count25_careful.csv"
filename = "real_data_count25_careful.h5"
construct_vectors(sns,filename,metafile)