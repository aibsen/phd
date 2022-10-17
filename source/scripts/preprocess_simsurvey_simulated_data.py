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

runs = 10
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

def create_gp_interpolated_vectors(data_fn, meta_fn, output_fn):

    data, metadata = load_data(data_fn, meta_fn)
    id_list = list(data.object_id.unique())
    # id_list_short = id_list[:5]
# sn_short = sn[sn.object_id.isin(id_list_short)]

    X, id_list, tags = preprocess_data_utils.generate_gp_all_objects(id_list,data,metadata,timesteps=128)
    flux_to_mag = lambda f: 30-2.5*math.log10(f)
    f = np.vectorize(flux_to_mag)
    X = f(X)

    dataset = {
        'X':X,
        'Y':tags,
        'ids':id_list
    }

    preprocess_data_utils.save_vectors(dataset,interpolated_data_dir+output_fn)
    # fluxerr_to_sigmag = lambda ferr,f: np.sqrt(np.abs(2.5/math.log(10)*(ferr/f)))
    # x = x[x.flux>0]
    # x['magpsf'] = [flux_to_mag(f) for f in x.flux]
    # x['sigmagpsf'] = [fluxerr_to_sigmag(f_err,f) for f,f_err in zip(x.flux,x.flux_err)]

def create_uneven_vectors(data_fn, meta_fn, output_fn):
    data, metadata = load_data(data_fn, meta_fn)
    # print(data.object_id.unique().shape)
    X, id_list, tags = preprocess_data_utils.create_uneven_vectors(data, metadata)

    dataset = {
        'X':X,
        'Y':tags,
        'ids':id_list
    }
    
    preprocess_data_utils.save_vectors(dataset,interpolated_data_dir+output_fn)

sn_f = 'simsurvey_sn4_balanced.csv'
sn_m_f = 'simsurvey_sn4_metadata_balanced.csv'

create_uneven_vectors(sn_f,sn_m_f,' simsurvey_data_balanced_4_mag_uneven.h5')
# sn = pd.read_csv(sn_f)
# sn_m = pd.read_csv(sn_m_f)
# sn_0 = sn[sn.object_id == 2900]
# print(sn_0)


# x = preprocess_data_utils.generate_gp_single_event(sn_0,timesteps=64)
# print(x)

# plot_utils.plot_raw_and_interpolated_lcs(sn_0,x)

# flux_to_mag = lambda f: 30-2.5*math.log10(f)
# fluxerr_to_sigmag = lambda ferr,f: np.sqrt(np.abs(2.5/math.log(10)*(ferr/f)))
# x = x[x.flux>0]
# x['magpsf'] = [flux_to_mag(f) for f in x.flux]
# x['sigmagpsf'] = [fluxerr_to_sigmag(f_err,f) for f,f_err in zip(x.flux,x.flux_err)]


# plot_utils.plot_raw_and_interpolated_lcs(sn_0,x,units='mag')


# id_list = list(sn.object_id.unique())
# id_list_short = id_list[:5]
# sn_short = sn[sn.object_id.isin(id_list_short)]
# create_gp_interpolated_vectors(sn, sn_m, 'simsurvey_data_balanced_4_mag_gp.h5')
# create_linearly_interpolated_vectors('simsurvey_sn4_balanced.csv','simsurvey_sn4_metadata_balanced.csv', 'simsurvey_data_balanced_4_mag_linear.h5')
# df = preprocess_data_utils.generate_gp_all_objects(id_list_short,sn_short,sn_m,timesteps=128)
# print(df)