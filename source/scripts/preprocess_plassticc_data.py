import numpy as np
import pandas as pd

import h5py
import gc
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *

plasticc_data_dir = "../../data/plasticc/" 
train_metadata_file = plasticc_data_dir+"csvs/plasticc_train_metadata.csv"
train_data_file = plasticc_data_dir+"csvs/plasticc_train_lightcurves.csv"

dataset_file = plasticc_data_dir+"plasticc_train_dataset.h5"


def create_interpolated_train_vectors():
        #get tags
        metadata = pd.read_csv(train_metadata_file)
        data = pd.read_csv(train_data_file)

        tags = metadata[["object_id","target"]]
        
        # X,obj_ids,Y = 
        obj_ids = create_interpolated_vectors_plasticc(data,128)
        print(tags[tags.object_id.isin(obj_ids)])

        # # write to file
        # if i == 1: #create dataset if it's the 1st chunk
        #     hf.create_dataset('X', data=X, compression="gzip", chunks=True, maxshape=(None,None,None)) 
        #     hf.create_dataset('Y', data=Y, compression="gzip", chunks=True, maxshape=(None,)) 
        #     hf.create_dataset('ids', data=obj_ids, compression="gzip", chunks=True, maxshape=(None,)) 
        # else: #resize the dataset otherwise
        #     hf["X"].resize((hf["X"].shape[0] + X.shape[0]), axis = 0)
        #     hf["X"][-X.shape[0]:] = X

        #     hf["Y"].resize((hf["Y"].shape[0] + Y.shape[0]), axis = 0)
        #     hf["Y"][-Y.shape[0]:] = Y

        #     hf["ids"].resize((hf["ids"].shape[0] + obj_ids.shape[0]), axis = 0)
        #     hf["ids"][-obj_ids.shape[0]:] = obj_ids




create_interpolated_train_vectors()