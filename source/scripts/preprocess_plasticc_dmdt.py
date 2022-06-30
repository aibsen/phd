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
import math

from preprocess_data_utils import *

plasticc_data_dir = "../../data/plasticc/csvs/" 
plasticc_processed_data_dir = "../../data/plasticc/dmdt/training/" 

# train_data_file = plasticc_data_dir+'plasticc_train_lightcurves.csv'
# train_metadata_file = plasticc_data_dir+"plasticc_train_metadata.csv"

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

def binning_strategy(code, resolution, min_max):
    range_stats = min_max.describe()
    # print(range_stats)
    max_dt = range_stats['dt']['max']
    # max_dt = 1.09406460e+03
    # max_dm = 3.58219712e+06
    max_dm = range_stats['dm']['max']
    max_dt_log = np.log10(max_dt)

    print(code)
    #binning strategies for dm bins: 0 means small changes are well represented, 1 means big changes are well represented 
    if code[0] == '0':
        max_dm_log = np.log10(max_dm)
        dm_bins = np.logspace(0,max_dm_log,int(resolution/2),base=10)
        dm_bins_not = np.sort(dm_bins*(-1))
        dm_bins = np.append(0,dm_bins)
        dm_bins = np.append(dm_bins_not,dm_bins)
        print(dm_bins)

    elif code[0] == '1':
        max_dm_percentile = min_max.dm.quantile(0.75)
        max_dm_log = np.log10(max_dm_percentile)
        dm_bins = np.logspace(1,max_dm_log,int(resolution/2)+1,base=10)
        dm_bins = np.sort(np.abs(dm_bins-max_dm_percentile))
        dm_bins_not = np.sort(dm_bins*(-1))
        dm_bins = np.append(0,dm_bins[1:])
        dm_bins = np.append(dm_bins_not[:-1],dm_bins)


    elif code[0]=='2': # linspace
        dm_bins = np.linspace(0,max_dm,int(resolution/2))
        dm_bins_not = np.sort(dm_bins*(-1))
        dm_bins = np.append(0,dm_bins)
        dm_bins = np.append(dm_bins_not,dm_bins)

    elif code[0]=='3': #logspace, but only up to 75 md percentile
        max_dm_percentile = min_max.dm.quantile(0.75)
        max_dm_log = np.log10(max_dm_percentile)
        dm_bins = np.logspace(0,max_dm_log,int(resolution/2),base=10)
        dm_bins_not = np.sort(dm_bins*(-1))
        dm_bins = np.append(0,dm_bins)
        dm_bins = np.append(dm_bins_not,dm_bins)

    elif code[0]=='4': #logspace, but only up to 75 md percentile
        min_dm_percentile = min_max.dm.quantile(0.1)
        max_dm_log = np.log10(max_dm)
        min_dm_log = np.log10(min_dm_percentile)
        dm_bins = np.logspace(min_dm_log,max_dm_log,int(resolution/2),base=10)
        dm_bins_not = np.sort(dm_bins*(-1))
        dm_bins = np.append(0,dm_bins)
        dm_bins = np.append(dm_bins_not,dm_bins)

    elif code[0]=='5': #logspace, but only up to 75 md percentile
        max_dm_log = np.log10(max_dm+100)
        dm_bins = np.logspace(0,max_dm_log,int(resolution/2),base=10)
        dm_bins_not = np.sort(dm_bins*(-1))
        dm_bins = np.append(0,dm_bins)
        dm_bins = np.append(dm_bins_not,dm_bins)

    elif code[0]=='6':
        # max_flux_delta = data["flux"].max()-data["flux"].min()
        # max_mjd_delta = data["mjd"].max()-data["mjd"].min()

        dm_bins = [-max_dm,-10000,-5000,-3000,-1000,-800,-600,-500,-300,-200,
            -100,-80,-60,-50,-40,-30,-20,-10,-5,-2,0,2,5,10,20,30,40,50,60,80,
            100,200,300,500,600,800,1000,3000,5000,10000,max_dm]

    elif code[0] == '7':
        max_dm_log = np.log10(max_dm)
        dm_bins = np.logspace(1,max_dm_log,int(resolution/2),base=10)
        dm_bins_not = np.sort(dm_bins*(-1))
        dm_bins = np.append(0,dm_bins)
        dm_bins = np.append(dm_bins_not,dm_bins)
        print(dm_bins)


    elif code[0] == '8':
        max_dm_log = np.log10(max_dm)
        dm_bins = np.logspace(2,max_dm_log,int(resolution/2),base=10)
        dm_bins_not = np.sort(dm_bins*(-1))
        dm_bins = np.append(0,dm_bins)
        dm_bins = np.append(dm_bins_not,dm_bins)


    if code[1] == '0':
        dt_bins = np.logspace(0,max_dt_log,resolution+1,base=10)
        print(dt_bins)
        
    elif code[1] == '1':
        dt_bins = np.logspace(1,max_dt_log,resolution+1,base=10)

    elif code[1] == '2':
        dt_bins = np.linspace(1,max_dt,resolution+1)

    elif code[1] == '3':
        max_dt_percentile = min_max.dt.quantile(0.75)
        max_dt_log = np.log10(max_dt_percentile)
        dt_bins = np.logspace(0,max_dt_log,resolution+1,base=10)
        print(dt_bins)
    elif code[1]=='6':
        dt_bins =  np.arange(0, max_dt,max_dt/41)

    # print(dm_bins)
    # print(dt_bins)
    return dm_bins, dt_bins


def create_DMDTS(data, n_objs,base=10,resolution=24, binning_strategy_code='00'):#fix base so it can be modified, currently it only works with 10
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

    dm_bins, dt_bins = binning_strategy(binning_strategy_code, resolution, min_max)

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


def create_test_set(metadata_file, data_file_template, output_fname, code='00'):
    
    n_file = [n for n in range(4,12)]
    metadata = pd.read_csv(metadata_file)
    print('\n{:,.0f} total objects in test set'.format(len(metadata.object_id.unique())))
    
    for i in n_file:
        f_name = data_file_template+'{}.csv'.format(i)
        print('\nfile {}/11'.format(i))
        data = pd.read_csv(f_name)
        print(len(data.object_id.unique()))
        
        obj_ids = data.object_id.unique()
        n_objs = len(obj_ids)
        relevant = metadata[metadata.object_id.isin(obj_ids)]
        relevant = relevant[relevant.true_target<900]
        data = data[data.object_id.isin(relevant.object_id)]
        targets = [plasticc_types.index(tag) for tag in relevant['true_target'].values]
        print(len(data.object_id.unique()))
        print(len(relevant.object_id.unique()))
        print('{:,.0f} objects'.format(n_objs))
        print('creating DMDTs ...')
        obj_ids = data.object_id.unique()
        n_objs = len(obj_ids)
        x = create_DMDTS(data,n_objs)
        print('almost done with {}'.format(i))
        print('saving ...')
        dataset = {'X':x,'ids':obj_ids,'Y':targets}
        output_file = plasticc_data_dir+output_fname+'{}_no99.h5'.format(i)
        save_dmdts(dataset,output_file)
        print('done with {}!'.format(i))
        # break


def add_noise(data, metadata, times=0):

    #shift mjd by obj
    # metadata_copy = metadata.copy()
    # metadata_copy['random_shift'] = np.random.rand(metadata_copy.shape[0])*np.random.randint(1,100)
    # random_shift = metadata_copy[['object_id','random_shift']]

    #copy data, change ids enough so they don't overlap existing objects, add id per lc
    noisy_data = data.copy()
    # noisy_data = noisy_data.merge(random_shift,how='inner')
    noisy_data['object_id'] = noisy_data.object_id*1000+times
    noisy_data['object_id_p'] = noisy_data.object_id*10+noisy_data.passband
    # noisy_data['mjd'] = noisy_data.mjd + noisy_data.random_shift

    #drop 10% of the light curves for objects that have more than the average points per lightcurve
    # points_per_lc = noisy_data.groupby(["object_id_p"]).count()['mjd']
    # mean_points_per_lc = points_per_lc.mean()
    # noisy_data = noisy_data.merge(points_per_lc,how='inner', on='object_id_p', suffixes=('','_count'))
    # enough_points = noisy_data.mjd_count>=mean_points_per_lc
    # print(noisy_data)
    # print(enough_points)
    # points_to_drop = noisy_data[enough_points][['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected_bool','object_id_p']]
    # points_to_keep = noisy_data[~enough_points][['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected_bool','object_id_p']]
    # points_to_drop = points_to_drop.groupby(["object_id_p"]).sample(frac=0.8)
    # noisy_data = pd.concat([points_to_drop, points_to_keep])[['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected_bool']].sort_values(by='object_id')

    #add random noise proportional to the flux error
    random_sign = np.random.rand(noisy_data.shape[0])
    random_sign = [1 if r >=0.5 else -1 for r in random_sign]
    random_scale = np.random.randint(1,3, noisy_data.shape[0])
    random_value = np.random.rand(noisy_data.shape[0])
    random_error = random_value*random_sign
    # random_error = random_value*random_sign*random_scale
    noisy_data['flux_err'] = noisy_data.flux_err*random_error
    noisy_data['flux'] = noisy_data['flux'] + noisy_data.flux_err 

    return noisy_data

def degrade_training_data(data_file, metadata_file):
    data = pd.read_csv(data_file)
    metadata =pd.read_csv(metadata_file)
    noisy_data = add_noise(data,metadata)
    print(data.shape)
    print(metadata.shape)
    print(noisy_data.shape)
    metadata['object_id'] = metadata['object_id']*1000
    noisy_data.to_csv(plasticc_data_dir+'plasticc_train_lightcurves_noisy70.csv',sep=',',index=False)
    metadata.to_csv(plasticc_data_dir+'plasticc_train_metadata_noisy70.csv',sep=',',index=False)


def augment_training_data_bit_balanced(data_file,metadata_file):
    data = pd.read_csv(data_file)
    metadata =pd.read_csv(metadata_file)
    dfs = []
    mdfs = []

    group_by_target = metadata.groupby('target').count()
    small_group = list(group_by_target[group_by_target.object_id<300].index)
    mid_group = list(group_by_target[(group_by_target.object_id>=300) & (group_by_target.object_id<1000)].index)
    # bigger_group = list(group_by_target[(group_by_target.object_id>=300) & (group_by_target.object_id<1000)].index)
    bigger_group = list(group_by_target[group_by_target.object_id>=1000].index)

    #for the small group, augment 2 times, keep original
    small_metadata = metadata.copy()
    small_metadata = small_metadata[small_metadata.target.isin(small_group)]
    small_data = data[data.object_id.isin(small_metadata.object_id.unique())]
    small_dfs = [small_data]
    small_mdfs = [small_metadata]
    # print(small_data)
    print(small_group)
    for i in range(0,5):
        small_group_noisy = add_noise(small_data,small_metadata,times=i)
        new_small_metadata = small_metadata.copy()
        new_small_metadata['object_id'] = small_metadata['object_id']*1000+i
        small_dfs.append(small_group_noisy)
        small_mdfs.append(new_small_metadata)
    
    small_dfs = pd.concat(small_dfs).sort_values(by='object_id')
    small_mdfs = pd.concat(small_mdfs).sort_values(by='object_id')
    print(len(small_dfs.object_id.unique()))
    print(len(small_mdfs.object_id.unique()))
    print(len(small_mdfs.object_id))

    # for mid group, augment twice, keep original
    mid_metadata = metadata.copy()
    mid_metadata = mid_metadata[mid_metadata.target.isin(mid_group)]
    mid_data = data[data.object_id.isin(mid_metadata.object_id.unique())]
    mid_dfs = [mid_data]
    mid_mdfs = [mid_metadata]
    print(mid_group)
    for i in range(0,3):
        mid_group_noisy = add_noise(mid_data,mid_metadata,times=i)
        new_mid_metadata = mid_metadata.copy()
        new_mid_metadata['object_id'] = mid_metadata['object_id']*1000+i
        mid_dfs.append(mid_group_noisy)
        mid_mdfs.append(new_mid_metadata)
    mid_dfs = pd.concat(mid_dfs).sort_values(by='object_id')
    mid_mdfs = pd.concat(mid_mdfs).sort_values(by='object_id')
    print(len(mid_dfs.object_id.unique()))
    print(len(mid_mdfs.object_id.unique()))
    print(len(mid_mdfs.object_id))
   
   
    # for bigger group, augment once, keep original
    bigger_metadata = metadata.copy()
    bigger_metadata = bigger_metadata[bigger_metadata.target.isin(bigger_group)]
    bigger_data = data[data.object_id.isin(bigger_metadata.object_id.unique())]
    print(bigger_group)
    bigger_group_noisy = add_noise(bigger_data,bigger_metadata)
    new_bigger_metadata = bigger_metadata.copy()
    new_bigger_metadata['object_id'] = bigger_metadata['object_id']*1000
    bigger_dfs = pd.concat([bigger_data, bigger_group_noisy]).sort_values(by='object_id')
    bigger_mdfs = pd.concat([bigger_metadata, new_bigger_metadata]).sort_values(by='object_id')
    print(len(bigger_dfs.object_id.unique()))
    print(len(bigger_mdfs.object_id.unique()))
    print(len(bigger_mdfs.object_id))
    
    # # for biggest group, augment once
    # biggest_metadata = metadata.copy()
    # biggest_metadata = biggest_metadata[biggest_metadata.target.isin(biggest_group)]
    # biggest_data = data[data.object_id.isin(biggest_metadata.object_id.unique())]
    # print(biggest_group)
    # biggest_noisy_group = add_noise(biggest_data,biggest_metadata)
    # biggest_noisy_group['object_id'] = biggest_noisy_group['object_id']/1000
    # biggest_noisy_group['object_id'] = biggest_noisy_group['object_id'].astype(int)
    # biggest_dfs=biggest_noisy_group.sort_values(by='object_id')
    # biggest_mdfs=biggest_metadata.sort_values(by='object_id')
    # print(len(biggest_df.object_id.unique()))
    # print(len(biggest_mdf.object_id.unique()))
    # print(len(biggest_mdf.object_id))

    all_data = pd.concat([small_dfs,mid_dfs,bigger_dfs]).sort_values(by='object_id')
    all_metadata = pd.concat([small_mdfs,mid_mdfs,bigger_mdfs]).sort_values(by='object_id')

    # print(all_data.shape[0])
    # print(all_metadata.shape[0])
    # print(len(all_metadata.object_id.unique())==len(all_data.object_id.unique()) and len(all_metadata.object_id.unique())==len(all_metadata.object_id))

    all_data.to_csv(plasticc_data_dir+'plasticc_train_lightcurves_augmented_noise4.csv',sep=',',index=False)
    all_metadata.to_csv(plasticc_data_dir+'plasticc_train_metadata_augmented_noise4.csv',sep=',',index=False)

    # targets_to_augment = list(group_by_target[group_by_target.object_id<300].index)
    # meta_objects_to_augment = metadata[metadata.target.isin(targets_to_augment)]
    # ids_to_augment = meta_objects_to_augment.object_id.unique()
    # objects_to_augment = data[data.object_id.isin(ids_to_augment)]
    # noisy_objects = add_noise(objects_to_augment,meta_objects_to_augment)

    # print(objects_to_augment.shape)
    # print(data.shape)
    # print(noisy_objects.shape)

def augment_training_data_sorta_balanced(data_file,metadata_file):
    data = pd.read_csv(data_file)
    metadata =pd.read_csv(metadata_file)
    dfs = []
    mdfs = []

    group_by_target = metadata.groupby('target').count()
    small_group = list(group_by_target[group_by_target.object_id<300].index)
    # mid_group = list(group_by_target[(group_by_target.object_id>=300) & (group_by_target.object_id<1000)].index)
    # bigger_group = list(group_by_target[(group_by_target.object_id>=300) & (group_by_target.object_id<1000)].index)
    bigger_group = list(group_by_target[group_by_target.object_id>=300].index)

    #for the small group, augment 2 times, keep original
    small_metadata = metadata.copy()
    small_metadata = small_metadata[small_metadata.target.isin(small_group)]
    small_data = data[data.object_id.isin(small_metadata.object_id.unique())]
    small_dfs = [small_data]
    small_mdfs = [small_metadata]
    # print(small_data)
    print(small_group)
    for i in range(0,7):
        small_group_noisy = add_noise(small_data,small_metadata,times=i)
        new_small_metadata = small_metadata.copy()
        new_small_metadata['object_id'] = small_metadata['object_id']*1000+i
        small_dfs.append(small_group_noisy)
        small_mdfs.append(new_small_metadata)
    
    small_dfs = pd.concat(small_dfs).sort_values(by='object_id')
    small_mdfs = pd.concat(small_mdfs).sort_values(by='object_id')
    print(len(small_dfs.object_id.unique()))
    print(len(small_mdfs.object_id.unique()))
    print(len(small_mdfs.object_id))

    # for mid group, augment twice, keep original
    # mid_metadata = metadata.copy()
    # mid_metadata = mid_metadata[mid_metadata.target.isin(mid_group)]
    # mid_data = data[data.object_id.isin(mid_metadata.object_id.unique())]
    # mid_dfs = [mid_data]
    # mid_mdfs = [mid_metadata]
    # print(mid_group)
    # for i in range(0,3):
    #     mid_group_noisy = add_noise(mid_data,mid_metadata,times=i)
    #     new_mid_metadata = mid_metadata.copy()
    #     new_mid_metadata['object_id'] = mid_metadata['object_id']*1000+i
    #     mid_dfs.append(mid_group_noisy)
    #     mid_mdfs.append(new_mid_metadata)
    # mid_dfs = pd.concat(mid_dfs).sort_values(by='object_id')
    # mid_mdfs = pd.concat(mid_mdfs).sort_values(by='object_id')
    # print(len(mid_dfs.object_id.unique()))
    # print(len(mid_mdfs.object_id.unique()))
    # print(len(mid_mdfs.object_id))
   
   
    # for bigger group, augment once, keep original
    bigger_metadata = metadata.copy()
    bigger_metadata = bigger_metadata[bigger_metadata.target.isin(bigger_group)]
    bigger_data = data[data.object_id.isin(bigger_metadata.object_id.unique())]
    print(bigger_group)
    bigger_group_noisy = add_noise(bigger_data,bigger_metadata)
    new_bigger_metadata = bigger_metadata.copy()
    new_bigger_metadata['object_id'] = bigger_metadata['object_id']*1000
    bigger_dfs = pd.concat([bigger_data, bigger_group_noisy]).sort_values(by='object_id')
    bigger_mdfs = pd.concat([bigger_metadata, new_bigger_metadata]).sort_values(by='object_id')
    print(len(bigger_dfs.object_id.unique()))
    print(len(bigger_mdfs.object_id.unique()))
    print(len(bigger_mdfs.object_id))
    
    # # for biggest group, augment once
    # biggest_metadata = metadata.copy()
    # biggest_metadata = biggest_metadata[biggest_metadata.target.isin(biggest_group)]
    # biggest_data = data[data.object_id.isin(biggest_metadata.object_id.unique())]
    # print(biggest_group)
    # biggest_noisy_group = add_noise(biggest_data,biggest_metadata)
    # biggest_noisy_group['object_id'] = biggest_noisy_group['object_id']/1000
    # biggest_noisy_group['object_id'] = biggest_noisy_group['object_id'].astype(int)
    # biggest_dfs=biggest_noisy_group.sort_values(by='object_id')
    # biggest_mdfs=biggest_metadata.sort_values(by='object_id')
    # print(len(biggest_df.object_id.unique()))
    # print(len(biggest_mdf.object_id.unique()))
    # print(len(biggest_mdf.object_id))

    all_data = pd.concat([small_dfs,bigger_dfs]).sort_values(by='object_id')
    all_metadata = pd.concat([small_mdfs,bigger_mdfs]).sort_values(by='object_id')

    # print(all_data.shape[0])
    # print(all_metadata.shape[0])
    # print(len(all_metadata.object_id.unique())==len(all_data.object_id.unique()) and len(all_metadata.object_id.unique())==len(all_metadata.object_id))

    all_data.to_csv(plasticc_data_dir+'plasticc_train_lightcurves_augmented_noise7.csv',sep=',',index=False)
    all_metadata.to_csv(plasticc_data_dir+'plasticc_train_metadata_augmented_noise7.csv',sep=',',index=False)

    # targets_to_augment = list(group_by_target[group_by_target.object_id<300].index)
    # meta_objects_to_augment = metadata[metadata.target.isin(targets_to_augment)]
    # ids_to_augment = meta_objects_to_augment.object_id.unique()
    # objects_to_augment = data[data.object_id.isin(ids_to_augment)]
    # noisy_objects = add_noise(objects_to_augment,meta_objects_to_augment)

    # print(objects_to_augment.shape)
    # print(data.shape)
    # print(noisy_objects.shape)


def augment_training_data(data_file, metadata_file):
    data = pd.read_csv(data_file)
    print(data.shape)
    metadata =pd.read_csv(metadata_file)
    # print(metadata.keys())
    dfs = []
    mdfs = []
    for i in range(0,3):
        noisy_data = add_noise(data, metadata, times=i)
        dfs.append(noisy_data)
        new_meta = metadata.copy()
        new_meta['object_id'] = new_meta['object_id']*1000+i
        mdfs.append(new_meta)

        # break
    randint = np.random.randint(0,100)
    print(randint)
    meta=metadata.iloc[randint]
    meta0=mdfs[0].iloc[randint]
    meta1=mdfs[1].iloc[randint]
    meta2=mdfs[2].iloc[randint]
    print(meta.object_id)
    print(meta.target)
    print(meta0.object_id)
    print(meta0.target)
    print(meta1.object_id)
    print(meta1.target)
    print(meta2.object_id)
    print(meta2.target)

    datao=data[data.object_id==meta.object_id]
    data0=dfs[0][dfs[0].object_id==meta0.object_id]
    data1=dfs[1][dfs[1].object_id==meta1.object_id]
    data2=dfs[2][dfs[2].object_id==meta2.object_id]
    # print(meta0)
    # print(meta1)
    plt.scatter(datao.mjd,datao.flux)
    plt.scatter(data0.mjd,data0.flux)
    plt.scatter(data1.mjd,data1.flux)
    plt.scatter(data2.mjd,data2.flux)
    plt.show()
    # print(dfs)
    # print(mdfs)
    new_data = pd.concat(dfs)
    print(new_data.shape)
    new_metadata = pd.concat(mdfs)
    print(new_metadata.shape)
    print(new_metadata[new_metadata.object_id==meta0.object_id][['object_id','target']])
    print(new_metadata[new_metadata.object_id==meta1.object_id][['object_id','target']])
    print(new_metadata[new_metadata.object_id==meta2.object_id][['object_id','target']])
    new_data.to_csv(plasticc_data_dir+'plasticc_train_lightcurves_augmented_5.csv',sep=',',index=False)
    new_metadata.to_csv(plasticc_data_dir+'plasticc_train_metadata_augmented_5.csv',sep=',',index=False)
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

def create_training_set(code='00'):
    train_data_file = plasticc_data_dir+'plasticc_train_lightcurves.csv'
    # train_data_file = plasticc_data_dir+'plasticc_train_lightcurves_augmented_noise4.csv'
    train_metadata_file = plasticc_data_dir+"plasticc_train_metadata.csv"
    # train_metadata_file = plasticc_data_dir+"plasticc_train_metadata_augmented_noise4.csv"
    meta_data = pd.read_csv(train_metadata_file)
    data = pd.read_csv(train_data_file)
    obj_ids = data.object_id.unique()
    y = [plasticc_types.index(tag) for tag in meta_data[meta_data.object_id==obj_ids]['target'].values]
    n_objs = len(meta_data.object_id.unique())
    print(y[10000:10010])
    print(meta_data.iloc[10000:10010].target)
    for resolution in [32,40,48]:
        print('creating DMDTs ... x{}'.format(resolution))
        x = create_DMDTS(data, n_objs, resolution=resolution, binning_strategy_code=code)
        print('saving ...')
        data_set = {'X':x, 'ids':obj_ids, 'Y':y}
        output_fname = plasticc_processed_data_dir+'dmdts_training_{}x{}_b{}.h5'.format(resolution,resolution,code)
        save_dmdts(data_set, output_fname)
        print(output_fname)
        # # break



# train_data_file = plasticc_data_dir+'plasticc_train_lightcurves.csv'
# train_metadata_file = plasticc_data_dir+"plasticc_train_metadata.csv"
# augment_training_data_sorta_balanced(train_data_file,train_metadata_file)
# for code in ['00']:
    # create_training_set(code)

test_metadata_file =  plasticc_data_dir+'plasticc_test_metadata.csv'
test_data_file = plasticc_data_dir+'plasticc_test_set_batch.csv'
for resolution in [24]:
    code = '00'
    output_fname = 'dmdts_test_{}x{}_b{}'.format(resolution,resolution,code)
    create_test_set(test_metadata_file, test_data_file_template,output_fname,code)
# x, ids, y = create_training_set(train_metadata_file,train_data_file)
# data_set = {'X':x, 'ids':ids, 'Y':y}
# output_fname = plasticc_processed_data_dir+'dmdts_training_32x32.h5'
# save_dmdts(data_set, output_fname)
# print(output_fname)



# data = pd.read_csv(train_data_file)
# metadata =pd.read_csv(train_metadata_file)

# add_noise(data,metadata)

# augment_training_data(train_data_file, train_metadata_file)
# degrade_training_data(train_data_file, train_metadata_file)
# check(train_metadata_file)
# 
# augment_training_data_bit_balanced(train_data_file,train_metadata_file)

# 

# train_metadata_file = plasticc_data_dir+"plasticc_train_metadata_augmented_noise2.csv"
# metadata = pd.read_csv(train_metadata_file)
# n_objs = len(metadata.object_id.unique())
# print(n_objs)
# objs_per_class

# train_metadata_file = plasticc_data_dir+"plasticc_train_metadata_augmented_noise3.csv"