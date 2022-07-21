import numpy as np
import pandas as pd
# import torch
# import torch.nn as nn
# import time
import h5py
import gc
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *

plasticc_data_dir = "../../data/plasticc/" 
metadata_file = plasticc_data_dir+"raw/plasticc_test_metadata.csv"
metadata = pd.read_csv(metadata_file)
dataset_file = plasticc_data_dir+"plasticc_dataset.h5"

def sanity_check_plot(original_X, original_Y, original_id, r_X, r_Y, r_id):
    print(original_X)
    fig, ax = plt.subplots(1, 2,figsize=(20,5))
    #plot original light curve
    for p in np.arange(6):
        lc_orginal = original_X[original_X.passband == p]
        lc_t = lc_orginal.mjd
        lc_f = lc_orginal.flux
        ax[0].plot(lc_t,lc_f)

        lc_r_f = r_X[p,:]
        lc_r_t = np.arange(r_X.shape[1])
        ax[1].plot(lc_r_t,lc_r_f)

    original_title = "object {} of type {}".format(original_id, original_Y)
    ax[0].set(title=original_title,xlabel='mjd', ylabel='f')
    r_title = "object {} of type {}".format(r_id, r_Y)
    ax[1].set(title=r_title,xlabel='mjd', ylabel='f')
    plt.show()
    

#I'm picking only sn and retagging them from 0 to 5
relevant_metadata = retag_plasticc(metadata)
sn_ids = relevant_metadata["object_id"].unique()
with h5py.File(dataset_file, 'a') as hf:
    for i in np.arange(1,12):
        data_file = "../../data/plasticc/raw/plasticc_test_set_batch{}.csv".format(i)
        data_chunk = pd.read_csv(data_file)    
        
        #filter data to consider sn only
        data_chunk = data_chunk[data_chunk["object_id"].isin(sn_ids)]
        ids = data_chunk["object_id"].unique()
        #get tags
        chunk_metadata = relevant_metadata[relevant_metadata["object_id"].isin(ids)]
        tags = chunk_metadata[["object_id","true_target"]]
        #convert vectors
        # print(tags)
        # print(data_chunk)
        X,obj_ids,Y = create_interpolated_vectors_plasticc(data_chunk,tags,128)
        gc.collect()
        
        # print(X.shape)
        # print(obj_ids.size)
        # print(Y.size)

        #sanity check, is order mantained?
        random = np.random.randint(0,Y.size-1)
        r_id = obj_ids[random]
        r_Y = Y[random]
        r_X = X[random]
        original_X = data_chunk[data_chunk.object_id == r_id]
        original_meta = chunk_metadata[chunk_metadata.object_id == r_id]
        original_id = original_meta.object_id.values
        original_Y = original_meta.true_target.values
        sanity_check_plot(original_X, original_Y, original_id, r_X, r_Y, r_id)

        # write to file
        if i == 1: #create dataset if it's the 1st chunk
            hf.create_dataset('X', data=X, compression="gzip", chunks=True, maxshape=(None,None,None)) 
            hf.create_dataset('Y', data=Y, compression="gzip", chunks=True, maxshape=(None,)) 
            hf.create_dataset('ids', data=obj_ids, compression="gzip", chunks=True, maxshape=(None,)) 
        else: #resize the dataset otherwise
            hf["X"].resize((hf["X"].shape[0] + X.shape[0]), axis = 0)
            hf["X"][-X.shape[0]:] = X

            hf["Y"].resize((hf["Y"].shape[0] + Y.shape[0]), axis = 0)
            hf["Y"][-Y.shape[0]:] = Y

            hf["ids"].resize((hf["ids"].shape[0] + obj_ids.shape[0]), axis = 0)
            hf["ids"][-obj_ids.shape[0]:] = obj_ids




