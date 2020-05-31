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


data_all = []
def print_stats(data):
    print("number of events",len(data.id.unique()))
    # print("time_diff stats")
    # group_by_id = data.groupby(['id'])['time']
    # time_delta=group_by_id.agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    # time_delta["time_diff"]=time_delta.time_max-time_delta.time_min
    # print(time_delta.describe())
    # print(time_delta.median())
    # print("npoints stats")
    # group_by_id_band = data.groupby(['id','band'])['time'].agg(['count']).rename(columns = lambda x : 'time_' + x).reset_index()
    # print(group_by_id_band.describe())
    # print(group_by_id_band.median())
    print("flux stats")
    print(data.describe())
    print(sns.flux.median())

id_count=0
for type in types:
    data=[]
    print("\n")
    print("calculating stats for type ",type_names[type])

    for i in np.arange(9):
        print("reading file ",str(i))
        #load snIa simulated lightcurves
        file_str = train_data_dir+"lcs_"+type_names[type]+"_00000"+str(i)+".pkl"
        # print("to df")
        sns = pkl_to_df(file_str)
        # print("done to df")
        sns = sns[sns['flux']>=0]
        sns.loc[sns.band==0,'flux'] = flux_to_abmag(sns.loc[sns.band==0, 'flux'],zp=26.275)#26.275
        sns.loc[sns.band==1, 'flux'] = flux_to_abmag(sns.loc[sns.band==1, 'flux'],zp=26.325)#26.325
        if len(data) == 0:
            data = sns
        else:
            data = pd.concat([data,sns])

        id_count = int(data.id.tail(1).values+1)
        # print(data.head())
        print("id count",id_count)
    print_stats(data)


    if len(data_all)==0:
        data_all=data
    else:
        data_all=pd.concat([data_all,data])

print("overall stats")
print_stats(data_all)
