
import os, sys
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *
from datasets import LCs, CachedLCs 

fpath = "/home/ai/phd/data/plasticc/"
lc_length = 128

plasticc_train_classes = [90,67,52,42,62,95,15,64,88,92,65,16,53,6,991,992,993,994,995]

def preprocess_train_data():
    metadata_fname = "plasticc_train_metadata.csv"
    data_fname = "plasticc_train_lightcurves.csv"

    data = pd.read_csv(fpath+data_fname)
    metadata = pd.read_csv(fpath+metadata_fname)
    metadata.loc[:,"true_target"] = [plasticc_train_classes.index(tag) for tag in metadata["true_target"]]

    X, ids, Y = create_interpolated_vectors_plasticc(data, metadata,lc_length)
    dataset = {'X':X, 'ids':ids, 'Y':Y}

    output_fname = fpath+'plasticc_train_data.h5'
    save_vectors(dataset, output_fname)


def preprocess_test_data():
    metadata_fname = "plasticc_test_metadata.csv"
    metadata = pd.read_csv(fpath+metadata_fname)

    for i in np.arange(1,12):
        data_fname = "plasticc_test_set_batch{}.csv".format(i)
        data = pd.read_csv(fpath+data_fname)

        ids = data.object_id.unique()
        chunk_metadata = metadata[metadata.object_id.isin(ids)].copy()
        chunk_metadata.loc[:,"true_target"] = [plasticc_train_classes.index(tag) for tag in chunk_metadata["true_target"]]
        assert(len(ids)==chunk_metadata.shape[0])
        print(str(len(chunk_metadata))+" ids in chunk "+str(i))
        
        X, ids, Y = create_interpolated_vectors_plasticc(data, chunk_metadata,lc_length)
        dataset = {'X':X, 'ids':ids, 'Y':Y}
        output_fname = fpath+'plasticc_test_data_batch{}.h5'.format(i)
        save_vectors(dataset, output_fname)
        print(output_fname)



# preprocess_train_data()
preprocess_test_data()


# interpolated_dataset_filename=fpath+'plasticc_test_data_batch1.h5'
# interpolated_dataset = LCs(lc_length, interpolated_dataset_filename)
# dataset_length = len(interpolated_dataset)
# print(dataset_length)
# print(interpolated_dataset[100])



# lc_length_stats = data.groupby(["object_id","passband"]).count().describe()
# print(lc_length_stats)
#mean = 30, min = 2, max = 72, 75% = 45

# group_by_mjd = data.groupby(['object_id','passband'])['mjd'].agg(['min', 'max'])
# group_by_mjd['delta']= group_by_mjd['max'] - group_by_mjd['min']
# print(group_by_mjd['delta'].describe())
#mean = 875 days, min = 283, max = 1094, avg length como dos años 75% = 888.75
#esto quiere decir que darle un length de lc_length sería como tener un punto por semana

# print(time_delta.head())
# X,ids,Y = create_interpolated_vectors_plasticc()

#get one SNIa asnd plot it
#print(metadata[metadata['true_target']==90]['object_id'])
# one_lc = data[data['object_id']==745]
# one_lc_p = one_lc.groupby('passband')
# for p, lc in one_lc_p:
#     print(lc)
#     t = lc.mjd
#     f = lc.flux
#     plt.scatter(t,f)
# plt.show()
#get the same interpolated SNIa and plot it
# X, ids, Y = create_interpolated_vectors_plasticc(data, metadata,lc_length)
# # print(list(ids).index(745)) #it's lc number 3
# one_interp_lc = X[3][0:6] #fluxes only
# t = np.arange(lc_length)
# for p in np.arange(one_interp_lc.shape[0]):
#     f = one_interp_lc[p]
#     plt.scatter(t,f)
# plt.show()


