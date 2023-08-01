from cProfile import run
from hashlib import new
from statistics import mean
import numpy as np
import pandas as pd
import os, sys
import math
import matplotlib.pyplot as plt
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess_data_utils
import plot_utils
from utils import simsurvey_ztf_type_dict

runs = 4
mag_cut = 19
csv_data_dir = '../../data/ztf/csv/simsurvey/'
interpolated_data_dir = '../../data/ztf/training/linearly_interpolated/'




# (r=0, g=1)
def filter_types(): #use SNe-Ia (0), Ib/c(2), II(1) and SLSNe(3) only
    sn_f = csv_data_dir+'simsurvey_lcs_balanced.csv'
    sn_m_f = csv_data_dir+'simsurvey_metadata_balanced.csv'

    sn_m = pd.read_csv(sn_m_f)
    print(sn_m.object_id.unique().shape)
    print(sn_m.true_target.unique())
    sn_4_types = sn_m[sn_m.true_target.isin([0,3,4,5])]
    sn_4_types.loc[sn_4_types.true_target == 3,'true_target'] = 1
    sn_4_types.loc[sn_4_types.true_target == 4,'true_target'] = 2
    sn_4_types.loc[sn_4_types.true_target == 5,'true_target'] = 3

    print(sn_4_types.true_target.unique())
    sn = pd.read_csv(sn_f)
    print(sn.passband.unique())
    print(sn.shape)
    sn_4_lcs = sn[sn.object_id.isin(sn_4_types.object_id.unique())]
    print(sn_4_lcs.shape)
    print(sn_4_lcs.object_id.unique().shape)
    print(sn_4_types.object_id.unique().shape)

    print((sn_4_lcs.object_id.unique() == sn_4_types.object_id.unique()).all())

    sn_4_lcs.to_csv(csv_data_dir+'simsurvey_sn4_balanced.csv',index=False)
    sn_4_types.to_csv(csv_data_dir+'simsurvey_sn4_metadata_balanced.csv',index=False)

def load_data(data_fn, meta_fn):
    data_fn = csv_data_dir+data_fn
    meta_fn = csv_data_dir+meta_fn

    data = pd.read_csv(data_fn)
    metadata = pd.read_csv(meta_fn)

    return data, metadata


simsurvey_ztf_type_dict_4 = {
    '0':'Ia',
    '1':'IIP', 
    '2':'Ibc',
    '3':'SLSN',
    '4':'IIn'
}



def merge_files():
    datas = []
    metadatas = []

    for run_code in range(runs):
        print(run_code)
        print("")
        for k, type_name in simsurvey_ztf_type_dict_4.items():
            
            data_fn = csv_data_dir+"lcs_{}_test_{}.csv".format(type_name,run_code)
            data = pd.read_csv(data_fn)

            metadata_fn = csv_data_dir+"meta_{}_test_{}.csv".format(type_name,run_code)
            metadata = pd.read_csv(metadata_fn)

            print(type_name)
            print(metadata.shape)
            print("")
    
            # print(metadata)

            true_target = int(k) if int(k)<4 else 1 #IIP and IIn are clumped into one class
            metadata['true_target'] = np.full(metadata.shape[0],true_target)
            datas.append(data)
            metadatas.append(metadata)
        print("")

    simsurvey_lcs = pd.concat(datas, axis=0, ignore_index=True)
    simsurvey_meta = pd.concat(metadatas, axis=0, ignore_index=True)
    print(simsurvey_meta.shape)
    print(simsurvey_meta.object_id.unique().shape)
    print(simsurvey_lcs.shape)
    print(simsurvey_lcs.object_id.unique().shape)

    simsurvey_lcs.to_csv(csv_data_dir+"simsurvey_lcs_test.csv",index=False)
    simsurvey_meta.to_csv(csv_data_dir+"simsurvey_metadata_test.csv",index=False)



def create_linearly_interpolated_vectors(data_fn, meta_fn, output_fn):

    data, metadata = load_data(data_fn, meta_fn)

    X,ids = preprocess_data_utils.create_interpolated_vectors(data,128,n_channels=2)
    tags = metadata[metadata.object_id.isin(ids)].sort_values(['object_id'])
    # print(tags)
    Y = tags.true_target.values

    # print((tags['object_id'].values == ids).all())
    # print((tags['object_id'].values == ids))

    # print(X.shape)
    # print(ids.shape)
    # print(Y.shape)
    assert((tags['object_id'].values == ids).all())    
    dataset = {
        'X':X,
        'Y':Y,
        'ids':ids
    }

    preprocess_data_utils.save_vectors(dataset,interpolated_data_dir+output_fn)

def create_gp_interpolated_vectors(data_fn, meta_fn, output_fn, careful=True):

    data, metadata = load_data(data_fn, meta_fn)
    id_list = list(data.object_id.unique())
    # id_list_short = id_list[:5]
# sn_short = sn[sn.object_id.isin(id_list_short)]

    X, id_list, tags, lens = preprocess_data_utils.create_gp_interpolated_vectors(id_list,data,metadata,timesteps=128,var_length=careful)
    flux_to_mag = lambda f: 30-2.5*math.log10(f) if f!= 1 else 0.0
    f = np.vectorize(flux_to_mag)
    X = f(X)
    print(X[0:10])

    dataset = {
        'X':X,
        'Y':tags,
        'ids':id_list
    }

    if careful:
        dataset['lens'] = lens

    preprocess_data_utils.save_vectors(dataset,interpolated_data_dir+output_fn)
    # fluxerr_to_sigmag = lambda ferr,f: np.sqrt(np.abs(2.5/math.log(10)*(ferr/f)))
    # x = x[x.flux>0]
    # x['magpsf'] = [flux_to_mag(f) for f in x.flux]
    # x['sigmagpsf'] = [fluxerr_to_sigmag(f_err,f) for f,f_err in zip(x.flux,x.flux_err)]



def create_uneven_vectors(data_fn, meta_fn, output_fn):
    data, metadata = load_data(data_fn, meta_fn)
    # print(data.object_id.unique().shape)
    X, id_list, tags, l = preprocess_data_utils.create_uneven_vectors(data, metadata)

    dataset = {
        'X':X,
        'Y':tags,
        'ids':id_list,
        'lens': l
    }
    print(dataset.keys())
    print(X)
    x = X[0]
    print(id_list[0])
    print(tags[0])
    print(l[0])
    print(x)
    # r = np.where(x[1]==1,x)
    # print(r)
    plt.figure()
    plt.scatter(x[-1],x[0])
    plt.gca().invert_yaxis()
    plt.show()

def create_linearly_interpolated_vectors_careful(data_fn, meta_fn, output_fn, 
    obs_days=30, points_per_band=3,n_channels=2, lc_max_length=128):

    lcs, meta = load_data(data_fn,meta_fn)
    
    X=np.zeros((meta.shape[0],n_channels,lc_max_length))
    X_void=np.zeros((meta.shape[0],n_channels,lc_max_length))
    lens=np.zeros((meta.shape[0],))

    group_by_id = lcs.groupby(['object_id'])['mjd'].agg(['min','max']).rename(columns = lambda x : 'mjd_' + x).reset_index()    
    group_by_id["mjd_diff"]=group_by_id.mjd_max-group_by_id.mjd_min

    obids = meta.object_id.unique()
    print(group_by_id)

    for i,object_id in enumerate(obids):
        lc = lcs[lcs.object_id == object_id]
        lc = lc.sort_values(by=['mjd'])
        print(object_id)
        lc_length = group_by_id.loc[group_by_id.object_id == object_id, 'mjd_diff'].values[0]
        lc_start = group_by_id.loc[group_by_id.object_id == object_id, 'mjd_min'].values[0]
        lc_stop = group_by_id.loc[group_by_id.object_id == object_id, 'mjd_max'].values[0]
        # print(lc_start)
        # print(lc_stop)
        lc_length=int(np.floor(lc_length))
        lc_length = lc_length if lc_length<=lc_max_length else lc_max_length
        # print(lc_length)
        t = np.arange(lc_length)
        #scale time
        lc['scaled_mjd'] = (lc_length-1)*(lc.mjd-lc_start)/(lc_stop-lc_start)
        lc_r = lc[lc.passband == 0] 
        lc_g = lc[lc.passband == 1]
        X[i,0,0:lc_length] = np.interp(t,lc_r.scaled_mjd,lc_r.magpsf)        
        X[i,1,0:lc_length] = np.interp(t,lc_g.scaled_mjd,lc_g.magpsf)
        X_void[i,0,0:lc_length] = [np.abs(lc_r.scaled_mjd - t_i).min() for t_i in t]
        X_void[i,1,0:lc_length] = [np.abs(lc_g.scaled_mjd - t_i).min() for t_i in t]
        lens[i] = lc_length

    vectors = np.concatenate((X, X_void), axis=1)
    print(vectors.shape)
    Y = meta.true_target.values
    dataset = {
    'X':vectors,
    'Y':Y,
    'ids':obids,
    'lens':lens
    }

    print((meta['object_id'].values == obids).all())
    print((meta['object_id'].values == obids))

    assert((meta['object_id'].values == obids).all())    

    preprocess_data_utils.save_vectors(dataset,output_fn)

sn_f = 'simsurvey_lcs_balanced.csv'
sn_m_f = 'simsurvey_metadata_balanced.csv'

# sn_f = 'simsurvey_lcs_test.csv'
# sn_m_f = 'simsurvey_metadata_test.csv'

create_linearly_interpolated_vectors_careful(sn_f, sn_m_f, 'simsurvey_data_balanced_6_d.h5')
# create_linearly_interpolated_vectors_careful(sn_f, sn_m_f, 'simsurvey_test_linear_careful.h5')

# create_uneven_vectors(sn_f,sn_m_f,'simsurvey_data_balanced_4_mag_uneven.h5')
# create_uneven_vectors(sn_f,sn_m_f,'simsurvey_test_uneven_tnorm_backl.h5')
# create_gp_interpolated_vectors(sn_f, sn_m_f, 'simsurvey_test_gp_careful.h5')
# create_gp_interpolated_vectors(sn_f, sn_m_f, 'simsurvey_data_balanced_4_mag_gp_careful.h5')
# create_linearly_interpolated_vectors(sn_f, sn_m_f, 'simsurvey_test_linear.h5')
# df = preprocess_data_utils.generate_gp_all_objects(id_list_short,sn_short,sn_m,timesteps=128)
# print(df)


# def sanity_check():
#     data_fn = csv_data_dir+"simsurvey_lcs_20.csv"
#     metadata_fn = csv_data_dir+"simsurvey_metadata_20.csv"

#     data = pd.read_csv(data_fn)
#     metadata = pd.read_csv(metadata_fn)

#     print((data.object_id.unique() == metadata.object_id.unique()).all())
#     print(data.object_id.unique().shape == metadata.object_id.unique().shape)
#     print(data.object_id.unique().shape)
