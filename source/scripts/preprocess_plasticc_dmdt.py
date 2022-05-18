import numpy as np
import pandas as pd
# import torch
# import torch.nn as nn
# import time
import h5py
import gc
import os, sys
import matplotlib.pyplot as plt
import itertools as it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *
# from dmdts_per_band import DMDT_per_band, DMDTs_per_band
# from utils import load_data, random_objs_per_class, get_classes



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *

plasticc_data_dir = "../../data/plasticc/csvs/" 
plasticc_processed_data_dir = "../../data/plasticc/dmdt/" 

train_data_file = plasticc_data_dir+'plasticc_train_lightcurves.csv'
train_metadata_file = plasticc_data_dir+"plasticc_train_metadata.csv"

test_data_file_template = plasticc_data_dir+'plasticc_test_set_batch'
test_metadata_file = plasticc_data_dir+"plasticc_test_metadata.csv"



plasticc_types = [90,67,52,42,62,95,15,64,88,92,65,16,53,6,991,992,993,994]


def save_dmdts(dataset, output_fname):
    hf=h5py.File(output_fname,'w')
    print("writing X")
    hf.create_dataset('X',data=dataset['X'])
    print("writing ids")
    hf.create_dataset('ids',data=dataset['ids'],dtype='int64')
    print("writing Y")
    hf.create_dataset('Y',data=dataset['Y'])
    hf.close()

def create_training_set(metadata_file, data_file, base=10, binm=24, bint=24):#fix base so it can be modified, currently it only works with 10
    meta_data = pd.read_csv(metadata_file)
    n_objs = len(meta_data.object_id.unique())
    n_channels = 6
    data = pd.read_csv(data_file)
    data['ob_p']=data.object_id*10+data.passband
    
    #get dmdt bins
    group_by_mjd=data.groupby(['object_id'])['mjd'].agg(['min','max']).rename(columns = lambda x: 'mjd_' + x).reset_index()
    group_by_flux=data.groupby(['object_id'])['flux'].agg(['min','max']).rename(columns = lambda x: 'flux_' + x).reset_index()
    min_max = pd.merge(group_by_flux,group_by_mjd, how="left", on='object_id')
    min_max['dm'] = min_max['flux_max'] - min_max['flux_min']
    min_max['dt'] = min_max['mjd_max'] - min_max['mjd_min']
    range_stats = min_max.describe()
    
    max_dt = range_stats['dt']['max']
    max_dt_log = np.log10(max_dt)
    dt_bins = np.logspace(1,max_dt_log,33,base=base)

    max_dm = range_stats['dm']['max']
    max_dm_log = np.log10(max_dm)
    dm_bins = np.logspace(0,max_dm_log,11,base=base)
    dm_bins_not = np.sort(dm_bins*(-1))
    dm_bins = np.append(0,dm_bins)
    dm_bins = np.append(dm_bins_not,dm_bins)

    data['count'] = 1
    data['cc'] = data.groupby(['ob_p'])['count'].cumcount()
    unstack = data[['ob_p','mjd','flux','cc']].set_index(['ob_p','cc']).unstack()
    mjds = unstack['mjd'].values
    fluxes = unstack['flux'].values
    nan_masks = ~np.isnan(mjds)[:,:]
    n_lcs = mjds.shape[0]
    DMDTs = np.zeros((n_lcs,len(dm_bins)-1,len(dt_bins)-1))
    c=0
    for i in range(n_lcs):
        true_mjds = mjds[i][nan_masks[i]]
        true_fluxes = fluxes[i][nan_masks[i]]
        dms = np.clip([(y - x) for x, y in it.combinations(true_fluxes, 2)], dm_bins[0],dm_bins[-1])
        dts = np.clip([(y - x) for x, y in it.combinations(true_mjds, 2)], dt_bins[0],dt_bins[-1])
        # fig,ax = plt.subplots(1,2)
        DMDTs[i],_,_ = np.histogram2d(dms,dts,bins=[dm_bins,dt_bins])
        DMDTs[i] = np.floor(DMDTs[i]*255/len(dms) + 0.999)  
    
    DMDTs_per_band = DMDTs.reshape((n_objs,n_channels,(len(dm_bins)-1),(len(dt_bins)-1)))
    obj_ids = data.object_id.unique()
    targets = [plasticc_types.index(tag) for tag in meta_data[meta_data.object_id==obj_ids]['target'].values]
    print(DMDTs_per_band.shape)
    return DMDTs_per_band, obj_ids, targets


def create_DMDTS(data, n_objs,base=10):#fix base so it can be modified, currently it only works with 10
    n_channels = 6
    # data = pd.read_csv(data_file)
    data['ob_p']=data.object_id*10+data.passband
    
    #get dmdt bins
    group_by_mjd=data.groupby(['object_id'])['mjd'].agg(['min','max']).rename(columns = lambda x: 'mjd_' + x).reset_index()
    group_by_flux=data.groupby(['object_id'])['flux'].agg(['min','max']).rename(columns = lambda x: 'flux_' + x).reset_index()
    min_max = pd.merge(group_by_flux,group_by_mjd, how="left", on='object_id')
    min_max['dm'] = min_max['flux_max'] - min_max['flux_min']
    min_max['dt'] = min_max['mjd_max'] - min_max['mjd_min']
    range_stats = min_max.describe()
    
    max_dt = range_stats['dt']['max']
    max_dt_log = np.log10(max_dt)
    dt_bins = np.logspace(1,max_dt_log,25,base=base)

    max_dm = range_stats['dm']['max']
    max_dm_log = np.log10(max_dm)
    dm_bins = np.logspace(0,max_dm_log,12,base=base)
    dm_bins_not = np.sort(dm_bins*(-1))
    dm_bins = np.append(0,dm_bins)
    dm_bins = np.append(dm_bins_not,dm_bins)

    data['count'] = 1
    data['cc'] = data.groupby(['ob_p'])['count'].cumcount()
    unstack = data[['ob_p','mjd','flux','cc']].set_index(['ob_p','cc']).unstack()
    mjds = unstack['mjd'].values
    fluxes = unstack['flux'].values
    nan_masks = ~np.isnan(mjds)[:,:]
    n_lcs = mjds.shape[0]
    DMDTs = np.zeros((n_lcs,len(dt_bins)-1,len(dm_bins)-1))
    c=0
    for i in range(n_lcs):
        # if nan_masks[i].any():
        true_mjds = mjds[i][nan_masks[i]]
        true_fluxes = fluxes[i][nan_masks[i]]
        dms = np.clip([(y - x) for x, y in it.combinations(true_fluxes, 2)], dm_bins[0],dm_bins[-1])
        dts = np.clip([(y - x) for x, y in it.combinations(true_mjds, 2)], dt_bins[0],dt_bins[-1])
        # fig,ax = plt.subplots(1,2)
        p = len(dms)
        if p>0:
            DMDTs[i],_,_ = np.histogram2d(dms,dts,bins=[dm_bins,dt_bins])
            DMDTs[i] = np.floor(DMDTs[i]*255/p + 0.999)

    DMDTs_per_band = DMDTs.reshape((n_objs,n_channels,(len(dt_bins)-1),(len(dt_bins)-1)))

    return DMDTs_per_band

def create_test_set(metadata_file, data_file_template):
    
    n_file = [n for n in range(2,12)]
    metadata = pd.read_csv(metadata_file)
    print('\n{:,.0f} total objects in test set'.format(len(metadata.object_id.unique())))
    
    for i in n_file:
        f_name = data_file_template+'{}.csv'.format(i)
        print('\nfile {}/11'.format(i))
        data = pd.read_csv(f_name)
        obj_ids = data.object_id.unique()
        n_objs = len(obj_ids)
        relevant = metadata[metadata.object_id.isin(obj_ids)]
        targets = [plasticc_types.index(tag) for tag in relevant['true_target'].values]
        print('{:,.0f} objects'.format(n_objs))
        print('creating DMDTs ...')
        x = create_DMDTS(data,n_objs)
        print('almost done with {}'.format(i))
        print('saving ...')
        dataset = {'X':x,'ids':obj_ids,'Y':targets}
        output_fname = plasticc_data_dir+'dmdts_test{}.h5'.format(i)
        save_dmdts(dataset,output_fname)
        print('done with {}!'.format(i))
        # break



# create_test_set(test_metadata_file, test_data_file_template)
x, ids, y = create_training_set(train_metadata_file,train_data_file)
data_set = {'X':x, 'ids':ids, 'Y':y}
output_fname = plasticc_processed_data_dir+'dmdts_training_32x32.h5'
save_dmdts(data_set, output_fname)
print(output_fname)



