import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *

train_data_dir = "../../data/training/raw/"
types = np.arange(4)
type_names = ["Ia_salt2","Ibc_nugent","IIn_nugent","IIP_nugent"]

id_count = 0

data = []
tags = []
ids = []

for i in np.arange(4): #half the files
    for type in types:
        print("building vectors for file ",str(i), " of type ",str(type))
        #load snIa simulated lightcurves
        file_str = train_data_dir+"lcs_"+type_names[type]+"_00000"+str(i)+".pkl"
        sns = pkl_to_df(file_str, id_count)

        #this here is to transform to magnitudes
        #first drop all negative sort_values
        # print(sns.head())
        sns = sns[sns['flux']>=0]
        #now do transform
        # print(sns.head())
        sns.loc[sns.band==0,'flux'] = flux_to_abmag(sns.loc[sns.band==0, 'flux'],zp=26.275)#26.275 ?? esto está mal
        sns.loc[sns.band==1, 'flux'] = flux_to_abmag(sns.loc[sns.band==1, 'flux'],zp=26.325)#26.325
        #now ensure that all objects have two bands
        group_by_id_band = sns.groupby(['id','band'])['time'].agg(['count']).rename(columns = lambda x : 'time_' + x).reset_index()
        group_by_id_band = group_by_id_band.groupby(['id']).count()
        ids_enough_point_count = group_by_id_band[group_by_id_band.time_count==2]
        usable_ids = list(set(ids_enough_point_count.index.values))
        sns = sns[sns.id.isin(usable_ids)]
        # print(sns.shape)

        # t = type if type < 3 else 2 
        # t = 2 if type == 3 else type 
        # sns_tags = df_tags(sns, t)
        sns_tags = df_tags(sns, type)
        id_count = int(sns_tags.id.tail(1).values+1)

        print("shape of df ", sns.shape)
        print("shape of tags ",sns_tags.shape)
        X,id,Y = create_interpolated_vectors(sns,sns_tags,128,n_channels=4)
        print("shape of vectors", X.shape)
        print("shape of tags", Y.shape)
        if len(data) == 0:
            data = X
            tags = Y
            ids = id
        else:
            data = np.concatenate((data, X))
            tags = np.concatenate((tags, Y))
            ids = np.concatenate((ids, id))

        print("shape of concat vectors", data.shape)
        print("shape of concat tags", tags.shape)
#total_sns = pd.concat(data,ignore_index=True)
#total_tags = pd.concat(tags, ignore_index=True)
#print("total shape of vectors", total_sns.shape)
#print("building vectors for file", total_tags.shape)

dataset = {
    'X':data,
    'Y':tags,
    'ids':ids
}

save_vectors(dataset,"unbalanced_dataset_m_realzp_128_small.h5")
