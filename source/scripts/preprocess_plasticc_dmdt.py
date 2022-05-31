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
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *

plasticc_data_dir = "../../data/plasticc/csvs/" 
plasticc_processed_data_dir = "../../data/plasticc/dmdt/" 

# train_data_file = plasticc_data_dir+'plasticc_train_lightcurves.csv'
# train_metadata_file = plasticc_data_dir+"plasticc_train_metadata.csv"

# test_data_file_template = plasticc_data_dir+'plasticc_test_set_batch'
# test_metadata_file = plasticc_data_dir+"plasticc_test_metadata.csv"



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

# def create_training_set(metadata_file, data_file, base=10, binm=24, bint=24):#fix base so it can be modified, currently it only works with 10
#     meta_data = pd.read_csv(metadata_file)
#     n_objs = len(meta_data.object_id.unique())
#     n_channels = 6
#     data = pd.read_csv(data_file)
#     data['ob_p']=data.object_id*10+data.passband
    
#     #get dmdt bins
#     group_by_mjd=data.groupby(['object_id'])['mjd'].agg(['min','max']).rename(columns = lambda x: 'mjd_' + x).reset_index()
#     group_by_flux=data.groupby(['object_id'])['flux'].agg(['min','max']).rename(columns = lambda x: 'flux_' + x).reset_index()
#     min_max = pd.merge(group_by_flux,group_by_mjd, how="left", on='object_id')
#     min_max['dm'] = min_max['flux_max'] - min_max['flux_min']
#     min_max['dt'] = min_max['mjd_max'] - min_max['mjd_min']
#     range_stats = min_max.describe()
    
#     max_dt = range_stats['dt']['max']
#     max_dt_log = np.log10(max_dt)
#     dt_bins = np.logspace(1,max_dt_log,33,base=base)

#     max_dm = range_stats['dm']['max']
#     max_dm_log = np.log10(max_dm)
#     dm_bins = np.logspace(0,max_dm_log,16,base=base)
#     dm_bins_not = np.sort(dm_bins*(-1))
#     dm_bins = np.append(0,dm_bins)
#     dm_bins = np.append(dm_bins_not,dm_bins)

#     data['count'] = 1
#     data['cc'] = data.groupby(['ob_p'])['count'].cumcount()
#     unstack = data[['ob_p','mjd','flux','cc']].set_index(['ob_p','cc']).unstack()
#     mjds = unstack['mjd'].values
#     fluxes = unstack['flux'].values
#     nan_masks = ~np.isnan(mjds)[:,:]
#     n_lcs = mjds.shape[0]
#     DMDTs = np.zeros((n_lcs,len(dm_bins)-1,len(dt_bins)-1))
#     c=0
#     for i in range(n_lcs):
#         true_mjds = mjds[i][nan_masks[i]]
#         true_fluxes = fluxes[i][nan_masks[i]]
#         dms = np.clip([(y - x) for x, y in it.combinations(true_fluxes, 2)], dm_bins[0],dm_bins[-1])
#         dts = np.clip([(y - x) for x, y in it.combinations(true_mjds, 2)], dt_bins[0],dt_bins[-1])
#         # fig,ax = plt.subplots(1,2)
#         DMDTs[i],_,_ = np.histogram2d(dms,dts,bins=[dm_bins,dt_bins])
#         DMDTs[i] = np.floor(DMDTs[i]*255/len(dms) + 0.999)  
    
#     DMDTs_per_band = DMDTs.reshape((n_objs,n_channels,(len(dm_bins)-1),(len(dt_bins)-1)))
#     obj_ids = data.object_id.unique()
#     targets = [plasticc_types.index(tag) for tag in meta_data[meta_data.object_id==obj_ids]['target'].values]
#     print(DMDTs_per_band.shape)
#     return DMDTs_per_band, obj_ids, targets


def create_DMDTS(data, n_objs,base=10,resolution=24):#fix base so it can be modified, currently it only works with 10
    n_channels = 6
    # data = pd.read_csv(data_file)
    data['ob_p']=data.object_id*10+data.passband
    # print(data.shape)
    #get dmdt bins
    group_by_mjd=data.groupby(['object_id'])['mjd'].agg(['min','max']).rename(columns = lambda x: 'mjd_' + x).reset_index()
    group_by_flux=data.groupby(['object_id'])['flux'].agg(['min','max']).rename(columns = lambda x: 'flux_' + x).reset_index()
    min_max = pd.merge(group_by_flux,group_by_mjd, how="left", on='object_id')
    min_max['dm'] = min_max['flux_max'] - min_max['flux_min']
    min_max['dt'] = min_max['mjd_max'] - min_max['mjd_min']
    range_stats = min_max.describe()
    # print(range_stats)
    # print(resolution)
    # #binning strategy 000: base 10, time starts from 1 month logspace, so small changes and short times are well represented
    max_dt = range_stats['dt']['max']
    max_dt_log = np.log10(max_dt)
    dt_bins = np.logspace(1,max_dt_log,resolution+1,base=base)
    max_dm = range_stats['dm']['max']
    max_dm_log = np.log10(max_dm)
    dm_bins = np.logspace(0,max_dm_log,int(resolution/2),base=base)
    dm_bins_not = np.sort(dm_bins*(-1))
    dm_bins = np.append(0,dm_bins)
    dm_bins = np.append(dm_bins_not,dm_bins)

    # print(len(dm_bins))
    # print(len(dt_bins))

    #binning strategy 001: base 10, time starts from 1 day, logspace, so small changes and short times are well represented
    # max_dt = range_stats['dt']['max']
    # max_dt_log = np.log10(max_dt)
    # dt_bins = np.logspace(0,max_dt_log,resolution+1,base=base)
    # max_dm = range_stats['dm']['max']
    # max_dm_log = np.log10(max_dm)
    # dm_bins = np.logspace(0,max_dm_log,int(resolution/2),base=base)
    # dm_bins_not = np.sort(dm_bins*(-1))
    # dm_bins = np.append(0,dm_bins)
    # dm_bins = np.append(dm_bins_not,dm_bins)

    #binning strategy 011: base e, time starts from 1 day, so small changes and short times are well represented
    # max_dt = range_stats['dt']['max']
    # max_dt_log = np.log(max_dt)
    # dt_bins = np.logspace(0,max_dt_log,resolution+1,base=np.e)
    # max_dm = range_stats['dm']['max']
    # max_dm_log = np.log(max_dm)
    # dm_bins = np.logspace(0,max_dm_log,int(resolution/2),base=np.e)
    # dm_bins_not = np.sort(dm_bins*(-1))
    # dm_bins = np.append(0,dm_bins)
    # dm_bins = np.append(dm_bins_not,dm_bins)

    #binning strategy 010: base e, time starts from 1 month, so small changes and short times are well represented
    # max_dt = range_stats['dt']['max']
    # max_dt_log = np.log(max_dt)
    # dt_bins = np.logspace(1,max_dt_log,resolution+1,base=np.e)
    # max_dm = range_stats['dm']['max']
    # max_dm_log = np.log(max_dm)
    # dm_bins = np.logspace(0,max_dm_log,int(resolution/2),base=np.e)
    # dm_bins_not = np.sort(dm_bins*(-1))
    # dm_bins = np.append(0,dm_bins)
    # dm_bins = np.append(dm_bins_not,dm_bins)

    # binning strategy 1: 'mirror' dm logspace, so big changes are well represented
    #binning strategy 100: base 10, time starts from 1 month, so small changes and short times are well represented
    # max_dt = range_stats['dt']['max']
    # max_dt_log = np.log(max_dt)
    # dt_bins = np.logspace(1,max_dt_log,resolution+1,base=np.e)
    # max_dm = range_stats['dm']['max']
    # max_dm_log = np.log(max_dm)
    # dm_bins = np.logspace(0,max_dm_log,int(resolution/2),base=np.e)
    # dm_bins_not = np.sort(dm_bins*(-1))
    # dm_bins = np.append(0,dm_bins)
    # dm_bins = np.append(dm_bins_not,dm_bins)

    # max_dt = range_stats['dt']['max']
    # max_dt_log = np.log10(max_dt)
    # dt_bins = np.logspace(1,max_dt_log,resolution+1,base=base)
    # max_dm = range_stats['dm']['max']
    # max_dm_log = np.log10(max_dm)
    # dm_bins = np.logspace(0,max_dm_log,int(resolution/2),base=base)
    # dm_bins_not = np.sort(dm_bins*(-1))
    # dm_bins = np.append(0,dm_bins)
    # dm_bins = np.append(dm_bins_not,dm_bins)

    # binning strategy 3: 'mirror' dmdt logspace, so big changes are well represented both in small and big times

    # binning strategy 4: linear


    data['count'] = 1
    data['cc'] = data.groupby(['ob_p'])['count'].cumcount()
    unstack = data[['ob_p','mjd','flux','cc']].set_index(['ob_p','cc']).unstack()
    mjds = unstack['mjd'].values
    fluxes = unstack['flux'].values
    # print(mjds.shape)
    # print(fluxes.shape)
   
    nan_masks = ~np.isnan(mjds)[:,:]
    n_lcs = mjds.shape[0]
    DMDTs = np.zeros((n_lcs,len(dt_bins)-1,len(dm_bins)-1))
    print(DMDTs.shape)
    c=0
    for i in range(n_lcs):
    #     # if nan_masks[i].any():
        true_mjds = mjds[i][nan_masks[i]]
        true_fluxes = fluxes[i][nan_masks[i]]
        dms = np.clip([(y - x) for x, y in it.combinations(true_fluxes, 2)], dm_bins[0],dm_bins[-1])
        dts = np.clip([(y - x) for x, y in it.combinations(true_mjds, 2)], dt_bins[0],dt_bins[-1])
    #     # fig,ax = plt.subplots(1,2)
        p = len(dms)
        if p>0:
            DMDTs[i],_,_ = np.histogram2d(dms,dts,bins=[dm_bins,dt_bins])
            DMDTs[i] = np.floor(DMDTs[i]*255/p + 0.999)

    DMDTs_per_band = DMDTs.reshape((n_objs,n_channels,(len(dt_bins)-1),(len(dt_bins)-1)))
    print(DMDTs_per_band.shape)
    return DMDTs_per_band


def create_test_set(metadata_file, data_file_template, output_fname):
    
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
        output_file = plasticc_data_dir+output_fname+'{}.h5'.format(i)
        save_dmdts(dataset,output_file)
        print('done with {}!'.format(i))
        # break


def add_noise(data, metadata, times=0):

    #shift mjd by obj
    metadata['random_shift'] = np.random.rand(metadata.shape[0])*np.random.randint(1,100)
    random_shift = metadata[['object_id','random_shift']]

    #copy data, change ids enough so they don't overlap existing objects, add id per lc
    noisy_data = data.copy()
    noisy_data = noisy_data.merge(random_shift,how='inner')
    noisy_data['object_id'] = noisy_data.object_id*1000+times
    noisy_data['object_id_p'] = noisy_data.object_id+noisy_data.passband
    noisy_data['mjd'] = noisy_data.mjd + noisy_data.random_shift

    #drop 30% of the light curves for objects that have more than the average points per lightcurve
    points_per_lc = noisy_data.groupby(["object_id_p"]).count()['mjd']
    mean_points_per_lc = points_per_lc.mean()
    noisy_data = noisy_data.merge(points_per_lc,how='inner', on='object_id_p', suffixes=('','_count'))
    enough_points = noisy_data.mjd_count>=mean_points_per_lc
    points_to_drop = noisy_data[enough_points][['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected_bool','object_id_p']]
    points_to_keep = noisy_data[~enough_points][['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected_bool','object_id_p']]
    points_to_drop = points_to_drop.groupby(["object_id_p"]).sample(frac=0.7)
    noisy_data = pd.concat([points_to_drop, points_to_keep])[['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected_bool']].sort_index()

    #add random noise proportional to the flux error
    random_sign = np.random.rand(noisy_data.shape[0])
    random_sign = [1 if r >=0.5 else -1 for r in random_sign]
    random_scale = np.random.randint(1,6, noisy_data.shape[0])
    random_value = np.random.rand(noisy_data.shape[0])
    random_error = random_value*random_sign*random_scale
    noisy_data['flux_err'] = noisy_data.flux_err*random_error
    noisy_data['flux'] = noisy_data.flux_err + noisy_data['flux'] 

    return noisy_data

def augment_training_data(data_file, metadata_file):
    data = pd.read_csv(data_file)
    print(data.shape)
    metadata =pd.read_csv(metadata_file)
    # print(metadata.keys())
    dfs = [data]
    mdfs = [metadata]
    for i in range(0,3):
        noisy_data = add_noise(data, metadata, times=i)
        dfs.append(noisy_data)
        new_meta = metadata.copy()
        new_meta['object_id'] = new_meta['object_id']*1000+i
        mdfs.append(new_meta)

    new_data = pd.concat(dfs, ignore_index=True)
    new_metadata = pd.concat(mdfs, ignore_index=True)
    new_data.to_csv(plasticc_data_dir+'plasticc_train_lightcurves_augmented.csv',sep=',',index=False)
    new_metadata.to_csv(plasticc_data_dir+'plasticc_train_metadata_augmented.csv',sep=',',index=False)
    # print(new_data)


def check(metadata_file):
    metadata =pd.read_csv(metadata_file)
    ids = metadata.object_id.values
    unique_ids = metadata.object_id.unique()
    print(len(ids)== len(unique_ids))
    ids0 = list(ids*1000)
    # print(ids0)
    ids1 = list(ids*1000 + 1)
    # print(ids1)
    ids2 = list(ids*1000 +2)
    # print(ids2)
    # new_ids = ids0+list(ids)
    new_ids = ids0+ids1+ids2+list(ids)
    print(len(new_ids))
    print(len(set(new_ids)))
    # print(len(ids)*3)

def create_training_set():
    train_data_file = plasticc_data_dir+'plasticc_train_lightcurves_augmented.csv'
    train_metadata_file = plasticc_data_dir+"plasticc_train_metadata_augmented.csv"
    meta_data = pd.read_csv(train_metadata_file)
    data = pd.read_csv(train_data_file)
    obj_ids = data.object_id.unique()
    # print(len(obj_ids))
    # print(len(meta_data.object_id.unique()))
    # print(data.shape)
    y = [plasticc_types.index(tag) for tag in meta_data[meta_data.object_id==obj_ids]['target'].values]
    n_objs = len(meta_data.object_id.unique())

    for resolution in [32, 40, 48, 56, 64]:
        print('creating DMDTs ... x{}'.format(resolution))
        x = create_DMDTS(data, n_objs, resolution=resolution)
        print('saving ...')
        data_set = {'X':x, 'ids':obj_ids, 'Y':y}
        output_fname = plasticc_processed_data_dir+'dmdts_training_{}x{}_augmented_b000.h5'.format(resolution,resolution)
        save_dmdts(data_set, output_fname)
        print(output_fname)
        # break


create_training_set()


# create_test_set(test_metadata_file, test_data_file_template)
# x, ids, y = create_training_set(train_metadata_file,train_data_file)
# data_set = {'X':x, 'ids':ids, 'Y':y}
# output_fname = plasticc_processed_data_dir+'dmdts_training_32x32.h5'
# save_dmdts(data_set, output_fname)
# print(output_fname)


# train_data_file = plasticc_data_dir+'plasticc_train_lightcurves.csv'
# train_metadata_file = plasticc_data_dir+"plasticc_train_metadata.csv"
# data = pd.read_csv(train_data_file)
# metadata =pd.read_csv(train_metadata_file)

# add_noise(data,metadata)

# augment_training_data(train_data_file, train_metadata_file)
# check(train_metadata_file)

