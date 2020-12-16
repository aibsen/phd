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
    group_by_id = sns_cp.groupby(['id'])['time'].agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    group_by_id["time_diff"]=group_by_id.time_max-group_by_id.time_min
    
    #ensure there is at least 30 days of obsevations
    ids_enough_obs=group_by_id[group_by_id.time_diff>days_obs].id.values
    group_by_id = group_by_id[group_by_id.id.isin(ids_enough_obs)]
    sn_enough=sns[sns.id.isin(ids_enough_obs)]

    #ensure there is at least 3 points per band
    group_by_id_band = sn_enough.groupby(['id','band'])['time'].agg(['count']).rename(columns = lambda x : 'time_' + x).reset_index()
    #drop all lcs that have less than 3 points
    at_least_3 = group_by_id_band[group_by_id_band.time_count>=point_count]
    #drop also their companion band
    band_counts = at_least_3.groupby(['id']).count().reset_index()
    band_counts = band_counts[band_counts.time_count==2]
    sn_enough=sn_enough[sn_enough.id.isin(band_counts.id)]
    tags_enough = tags[tags.objid.isin(band_counts.id)]

    #coonvert bands to the code's sim uses (r=0, g=1)
    #instead of lasair's (g=1, r=2)
    sn_enough.loc[sn_enough.band==2,'band'] = 0

    count = 0
    max_length = group_by_id.time_diff.max()
    max_scaled_length = int(np.ceil(128*max_length/128))
    X=np.zeros((tags_enough.shape[0],4,max_scaled_length+2))

    obids = tags_enough.objid.unique()
    for n,objid in enumerate(obids):
        lc = sn_enough[sn_enough.id == objid]
        lc_r = lc[lc.band == 0] 
        lc_g = lc[lc.band == 1]
        # print(objid)
        lc_length = group_by_id.loc[group_by_id.id == objid, 'time_diff'].values[0]
        lc_start = group_by_id.loc[group_by_id.id == objid, 'time_min'].values[0]
        lc_stop = group_by_id.loc[group_by_id.id == objid, 'time_max'].values[0]
        scaled_lc_length=int(np.ceil(128*lc_length/128))
        lc_step = lc_length/scaled_lc_length
        new_x = np.arange(lc_start,lc_stop+1,lc_step)
        X[n,0,0:scaled_lc_length+2] = np.interp(new_x,lc_r.time, lc_r.flux)
        X[n,1,0:scaled_lc_length+2] = np.interp(new_x,lc_g.time, lc_g.flux)

        for i in range(scaled_lc_length+1):
            X[n,2,i] = np.abs(lc_r.time.values - new_x[i]).min()
            X[n,3,i] = np.abs(lc_g.time.values - new_x[i]).min()
            
    print(X.shape)
    print(tags_enough.id.unique().shape)
    print(tags_enough.tag.values.shape)

    dataset = {
    'X':X,
    'Y':tags_enough.tag.values,
    'ids':tags_enough.id.unique()
    }
    # print(dataset)

    if metafile:
        tags_enough.to_csv(meta_file)
    save_vectors(dataset,filename)


data_dir="../../data/testing/27-06-2020-sns/"
filename = "data_4_types.csv"
data = pd.read_csv(data_dir+filename)
metafile = "metadata_4_types.csv"
metadata = pd.read_csv(data_dir+metafile)
# print(data)
counts = [3,5,10,15,20,25,30,35,40]
for c in counts:
    filename = "real_data_30do_count{}.h5".format(c)
    print(filename)
    construct_vectors(data,metadata,filename,point_count=c)