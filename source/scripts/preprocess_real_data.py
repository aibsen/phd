from distutils.command.clean import clean
from fileinput import filename
from tokenize import group
from venv import create
import pandas as pd
import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess_data_utils
import matplotlib.pyplot as plt


# (r=0, g=1)

data_dir = "../../data/ztf/csv/tns/"

ztf_type_dict_4 = {
    '0':'SN-Ia',
    '1':'SN-II', 
    '2':'SN-Ib/c',
    '3':'SLSN',
}

ztf_names = [ztf_type_dict_4[k] for k in ztf_type_dict_4]





def at_leat_obs_days(lcs_fn, meta_fn, points_per_band=3, obs_days=30, n_channels = 2):
    
    #search for max brightness
    group_by_id_bright = lcs_fn.groupby(['object_id'])['magpsf'].agg(['min']).rename(columns = lambda x : 'max_magpsf').reset_index()
    merged = pd.merge(lcs_fn,group_by_id_bright,how='left', on='object_id')
    
    #search for time of max brightness
    time_max_mag = merged[merged.magpsf==merged.max_magpsf][['object_id','mjd']]
    merged = pd.merge(merged,time_max_mag,how='left', on='object_id',suffixes=('','_max'))
    
    #only keep points within 100 days of max brightness
    merged = merged[np.abs(merged.mjd_max-merged.mjd)<100]
    group_by_id = merged.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'mjd_' + x).reset_index()
    group_by_id["mjd_diff"]=group_by_id.mjd_max-group_by_id.mjd_min
    
    #use objs with at least obs_days observation days
    ids_enough_obs=group_by_id[(group_by_id.mjd_diff>=obs_days)].object_id
    group_by_id = group_by_id[group_by_id.object_id.isin(ids_enough_obs)]
    sn_enough=merged[merged.object_id.isin(ids_enough_obs)]

    #ensure there is at least 3 points per band
    group_by_id_band = sn_enough.groupby(['object_id','passband'])['mjd'].agg(['count']).rename(columns = lambda x : 'time_' + x).reset_index()
    #drop all lcs that have less than 3 points
    at_least_p = group_by_id_band[group_by_id_band.time_count>=points_per_band]
    #drop also their companion band
    band_counts = at_least_p.groupby(['object_id']).count().reset_index()
    band_counts = band_counts[band_counts.time_count==n_channels]
    sn_enough=sn_enough[sn_enough.object_id.isin(band_counts.object_id)]
    meta_enough = meta_fn[meta_fn.object_id.isin(band_counts.object_id)]
    group_by_id = group_by_id[group_by_id.object_id.isin(band_counts.object_id)]
    print(group_by_id)

    #return df with filtered lcs, filtered metadata and time-frame for each obj

    return sn_enough, meta_enough, group_by_id

def create_linearly_interpolated_vectors_careful(data_fn, meta_fn, output_fn, 
    obs_days=30, points_per_band=3,n_channels=2, lc_max_length=128):

    lcs = pd.read_csv(data_fn)
    meta = pd.read_csv(meta_fn)

    sn_enough, tags_enough, group_by_id = at_leat_obs_days(lcs, meta, points_per_band, obs_days, n_channels)
    
    X=np.zeros((tags_enough.shape[0],n_channels,lc_max_length))
    X_void=np.zeros((tags_enough.shape[0],n_channels,lc_max_length))
    lens=np.zeros((tags_enough.shape[0],))

    obids = tags_enough.object_id.unique()

    for i,object_id in enumerate(obids):
        lc = sn_enough[sn_enough.object_id == object_id]
        lc = lc.sort_values(by=['mjd'])
        # print(object_id)
        lc_length = group_by_id.loc[group_by_id.object_id == object_id, 'mjd_diff'].values[0]
        lc_start = group_by_id.loc[group_by_id.object_id == object_id, 'mjd_min'].values[0]
        lc_stop = group_by_id.loc[group_by_id.object_id == object_id, 'mjd_max'].values[0]
        # print(lc_start)
        # print(lc_stop)
        lc_length=int(np.floor(lc_length))
        lc_length = lc_length if lc_length<=lc_max_length else lc_max_length
        print(lc_length)
        t = np.arange(lc_length)
        #scale time
        lc['scaled_mjd'] = (lc_length-1)*(lc.mjd-lc_start)/(lc_stop-lc_start)
        lc_r = lc[lc.passband == 0] 
        lc_g = lc[lc.passband == 1]
        X[i,0,-lc_length:] = np.interp(t,lc_r.scaled_mjd,lc_r.magpsf)        
        X[i,1,-lc_length:] = np.interp(t,lc_g.scaled_mjd,lc_g.magpsf)
        X_void[i,0,-lc_length:] = [np.abs(lc_r.scaled_mjd - t_i).min() for t_i in t]
        X_void[i,1,-lc_length:] = [np.abs(lc_g.scaled_mjd - t_i).min() for t_i in t]
        lens[i] = lc_length
    

    vectors = np.concatenate((X, X_void), axis=1)
    print(vectors.shape)
    Y = tags_enough.true_target.values
    ids = tags_enough.object_id.unique()
    dataset = {
    'X':vectors,
    'Y':Y,
    'ids':ids,
    'lens':lens
    }

    print((tags_enough['object_id'].values == ids).all())
    print((tags_enough['object_id'].values == ids))

    assert((tags_enough['object_id'].values == ids).all())    

    preprocess_data_utils.save_vectors(dataset,output_fn)

def create_linearly_interpolated_vectors(lcs_fn,meta_fn,output,
    obs_days=30, points_per_band=3, n_channels=2, lc_max_length=128):

    lcs = pd.read_csv(lcs_fn)
    meta = pd.read_csv(meta_fn)

    sn_enough, tags_enough, group_by_id = at_leat_obs_days(lcs, meta, points_per_band, obs_days, n_channels)

    X, ids = preprocess_data_utils.create_interpolated_vectors(sn_enough,lc_max_length,n_channels)
    
    tags = meta[meta.object_id.isin(ids)].sort_values(['object_id'])
    print(tags)
    Y = tags.true_target.values

    Y = meta[meta.object_id.isin(ids)].true_target.values.astype("long")
    print(Y.shape)
    print(X.shape)
    print(ids.shape)

    print((tags['object_id'].values == ids).all())
    print((tags['object_id'].values == ids))

    print(X.shape)
    print(ids.shape)
    print(Y.shape)
    assert((tags['object_id'].values == ids).all())    

    dataset = {
        'X':X,
        'Y':Y,
        'ids':ids
    }

    # preprocess_data_utils.save_vectors(dataset,output)

def create_gp_interpolated_vectors(lcs_fn, meta_fn, output_fn,
    points_per_band=3, obs_days=30, n_channels = 2, lc_max_length = 128):
    
    lcs = pd.read_csv(lcs_fn)
    meta = pd.read_csv(meta_fn)
    print(lcs)

    sn_enough, tags_enough, group_by_id = at_leat_obs_days(lcs, meta, points_per_band, obs_days, n_channels)
    id_list = list(tags_enough.object_id.unique())

    mag_to_flux = lambda m: math.pow(10,(30-m)/2.5)
    sn_enough['flux'] = [mag_to_flux(m) for m in sn_enough.magpsf]
    sim_error = np.random.normal(0, 1, sn_enough.shape[0])
    sn_enough['flux_err'] = sim_error*sn_enough.flux
    print(sn_enough)

    X,id_list, tags = preprocess_data_utils.create_gp_interpolated_vectors(id_list,sn_enough,tags_enough,
        timesteps=lc_max_length)

    flux_to_mag = lambda f: 30-2.5*math.log10(f)
    f = np.vectorize(flux_to_mag)
    X = f(X)
    
    dataset = {
        'X':X,
        'Y':tags,
        'ids':id_list
    }

    preprocess_data_utils.save_vectors(dataset,output_fn)


def create_uneven_vectors(lcs_fn, meta_fn, output_fn,
    points_per_band=3, obs_days=30, n_channels=2, lc_max_length=128):

    lcs = pd.read_csv(lcs_fn)
    meta = pd.read_csv(meta_fn)

    sn_enough, tags_enough, group_by_id = at_leat_obs_days(lcs, meta, points_per_band, obs_days, n_channels)

    print(tags_enough)
    X, id_list, tags, l = preprocess_data_utils.create_uneven_vectors(sn_enough, tags_enough, mag=True)

    dataset = {
        'X':X,
        'Y':tags,
        'ids':id_list,
        'lens': l
    }

    preprocess_data_utils.save_vectors(dataset,output_fn)



# create_linearly_interpolated_vectors(data_dir+'mars_sn_lcs_4.csv', data_dir+'mars_sn_meta_4.csv', data_dir+'real_test_linear_3pb_30obsd.h5')
create_linearly_interpolated_vectors_careful(data_dir+'mars_sn_lcs_4.csv', data_dir+'mars_sn_meta_4.csv', data_dir+'real_test_linear_3pb_30obsd_careful.h5')
# create_gp_interpolated_vectors(data_dir+'mars_sn_lcs_4.csv', data_dir+'mars_sn_meta_4.csv', data_dir+'real_test_gp_3pb_30obsd_careful.h5')
# create_uneven_vectors(data_dir+'mars_sn_lcs_4.csv', data_dir+'mars_sn_meta_4.csv', data_dir+'real_test_uneven_3pb_30obsd.h5')







