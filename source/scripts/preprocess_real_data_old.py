from tokenize import group
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

def construct_vectors(sns,tags,filename,meta_file=None,days_obs=30,point_count=3):
    print('\n')
    print(len(sns))

    sns_cp = sns.copy()
    
    #only keep observation 100 days after peak 

    group_by_id = sns_cp.groupby(['object_id'])['magpsf'].agg(['min']).rename(columns = lambda x : 'max_magpsf').reset_index()
    merged = pd.merge(sns_cp, group_by_id, how = 'left', on = 'object_id')
    print(merged)

    max_t = merged[merged.magpsf==merged.max_magpsf][['object_id','mjd']].rename(columns = {'mjd': 'max_mag_mjd'})

    merged = pd.merge(merged,max_t, how='left', on='object_id')
    print(merged)
    merged = merged[(merged.mjd>merged.max_mag_mjd-30) & (merged.mjd< merged.max_mag_mjd+100)]
    print(merged)

    group_by_id = merged.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    group_by_id["time_diff"]=group_by_id.time_max-group_by_id.time_min
    print(group_by_id.time_diff.max())
    print(group_by_id)

    #ensure there is at least 30 days of obsevations and less than 130?
    ids_enough_obs=group_by_id[(group_by_id.time_diff>days_obs) & (group_by_id.time_diff<131)].object_id.values
    # print(ids_enough_obs.shape)
    # ids_enough_obs=group_by_id[group_by_id.time_diff>days_obs].object_id.values
    group_by_id = group_by_id[group_by_id.object_id.isin(ids_enough_obs)]
    print(group_by_id)
    sn_enough=sns[sns.object_id.isin(ids_enough_obs)]

    #ensure there is at least 3 points per band
    group_by_id_band = sn_enough.groupby(['object_id','passband'])['mjd'].agg(['count']).rename(columns = lambda x : 'time_' + x).reset_index()
    #drop all lcs that have less than 3 points
    at_least_3 = group_by_id_band[group_by_id_band.time_count>=point_count]
    #drop also their companion band
    band_counts = at_least_3.groupby(['object_id']).count().reset_index()
    band_counts = band_counts[band_counts.time_count==2]
    sn_enough=sn_enough[sn_enough.object_id.isin(band_counts.object_id)]
    tags_enough = tags[tags.object_id.isin(band_counts.object_id)]
    
    count = 0
    max_length = group_by_id.time_diff.max()
    max_scaled_length = int(np.floor(128*max_length/128))
    print(max_length)
    print(max_scaled_length)

    X=np.zeros((tags_enough.shape[0],4,max_scaled_length))
    lens = np.zeros((tags_enough.shape[0],))
    obids = tags_enough.object_id.unique()
    print(obids.shape)

    for n,object_id in enumerate(obids):
        lc = sn_enough[sn_enough.object_id == object_id]
        lc_r = lc[lc.passband == 0] 
        lc_g = lc[lc.passband == 1]
        print(object_id)
        lc_length = group_by_id.loc[group_by_id.object_id == object_id, 'time_diff'].values[0]
        lc_start = group_by_id.loc[group_by_id.object_id == object_id, 'time_min'].values[0]
        lc_stop = group_by_id.loc[group_by_id.object_id == object_id, 'time_max'].values[0]
        scaled_lc_length=int(np.floor(128*lc_length/128))
        lc_step = lc_length/scaled_lc_length
        new_x = np.arange(lc_start,lc_stop+1,lc_step)
    #     print(new_x)
        print(new_x.shape)
        print(lc_r.mjd.shape)
        X[n,0,0:new_x.shape[0]] = np.interp(new_x,lc_r.mjd, lc_r.magpsf)
        X[n,1,0:new_x.shape[0]] = np.interp(new_x,lc_g.mjd, lc_g.magpsf)
        lens[n]=new_x.shape[0]

        for i in range(scaled_lc_length+1):
            X[n,2,i] = np.abs(lc_r.mjd.values - new_x[i]).min()
            X[n,3,i] = np.abs(lc_g.mjd.values - new_x[i]).min()
            
    print(X.shape)
    print(lens)
    # print(tags_enough.object_id.unique().shape)
    # print(tags_enough.true_target.values.shape)

    dataset = {
    'X':X,
    'Y':tags_enough.true_target.values,
    'ids':tags_enough.object_id.unique(),
    'lens':lens
    }
    print(dataset)

    # # if metafile:
    # #     tags_enough.to_csv(meta_file)
    save_vectors(dataset,filename)


data_dir="../../data/ztf/csv/tns/"
filename = "mars_sn_lcs_clean.csv"
data = pd.read_csv(data_dir+filename)
metafile = "mars_sn_metadata_clean.csv"
metadata = pd.read_csv(data_dir+metafile)
# print(data)
# counts = [3,5,10,15,20,25,30,35,40]
# for c in counts:
c=30
filename = "real_data_count_30.h5"
    # print(filename)
construct_vectors(data,metadata,filename,point_count=c)