import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *

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


for type, sn in enumerate(sns_list):
    # print("type ", type)
    sn_tags = df_tags(sn, type)
    if len(tags)==0:
         tags=sn_tags
    else:
        tags=pd.concat([tags,sn_tags])

def construct_vectors(sns,filename,meta_file=None,percentile=None):
    print('\n')
    print(len(sns))

    lc_length,ids_enough_obs=check_lc_length(sns,percentile)
    print("constructing vectors of real data for lcs with at least {} days observations ".format(int(lc_length)))
    print(len(ids_enough_obs))
    sn_enough=sns[sns.id.isin(ids_enough_obs)]
    tags_enough = tags[tags.id.isin(ids_enough_obs)]
    #coonvert bands to the code's sim uses (r=0, g=1)
    #instead of lasair's (g=1, r=2)
    sn_enough.loc[sn_enough.band==2,'band'] = 0
    print(tags_enough.head())
    tags_enough["old_id"]=tags_enough['id']
    print(tags_enough.head())
    for c,i in enumerate(tags_enough.old_id.unique()):
        tags_enough.loc[tags_enough.old_id==i,'id']=c
        sn_enough.loc[sn_enough.id==i,'id'] = c
    print(tags_enough.head())

    scaled_lc_length = int(np.ceil(128*lc_length/328))

    if metafile:
        tags_enough.to_csv(metafile.format(str(percentile),scaled_lc_length))
    print(sn_enough)
    X, obids, Y = create_interpolated_vectors(sn_enough, tags_enough, scaled_lc_length, dtype='real',n_channels=4)
    # break
    print("data shape ", X.dtype)
    print("tags shape ",Y.dtype)
    print("ids shape ",obids.dtype,"\n")

    print(obids.astype(int))
    dataset = {
    'X':X,
    'Y':Y,
    'ids':obids.astype(int)
    }
    save_vectors(dataset,filename.format(percentile,scaled_lc_length))




percentiles = ['25%','50%','75%']
for p in percentiles:
    metafile="real_data_tags_{}p_{}.csv"
    filename = "real_data_{}p_{}.h5"
    construct_vectors(sns,filename,metafile,p)
    #
