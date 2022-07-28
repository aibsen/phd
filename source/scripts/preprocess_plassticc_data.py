import numpy as np
import pandas as pd

import h5py
import gc
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import *

plasticc_data_dir = "../../data/plasticc/" 

train_metadata_file = plasticc_data_dir+"csvs/plasticc_new10kshuffle_train_metadata.csv"
train_data_file = plasticc_data_dir+"csvs/plasticc_new10kshuffle_train_data.csv"

test_metadata_file = plasticc_data_dir+"csvs/plasticc_new10kshuffle_test_metadata.csv"
test_data_file_template = plasticc_data_dir+"csvs/plasticc_new10kshuffle_test_set_batch"

plasticc_classes = [90,67,52,42,62,95,15,64,88,92,65,16,53,6,991,992,993,994,995]


def preprocess_train_data():
        metadata = pd.read_csv(train_metadata_file)
        data = pd.read_csv(train_data_file)

        X, obj_ids = create_interpolated_vectors_plasticc(data,128)

        tags = metadata[["object_id", "true_target"]]
        tags = tags[tags.object_id.isin(obj_ids)]
        tags.loc[:,"true_target"] = [plasticc_classes.index(tag) for tag in metadata["true_target"]]
        Y = tags.true_target.values

        dataset = {'X':X, 'ids':obj_ids, 'Y':Y}
        output_fname = plasticc_data_dir+"interpolated/training/plasticc_train_data_balanced.h5"
        save_vectors(dataset, output_fname)

def preprocess_test_data():
        metadata = pd.read_csv(test_metadata_file)

        for i in np.arange(3,12):
                data_fname = "csvs/plasticc_new10kshuffle_test_set_batch{}.csv".format(i)
                data = pd.read_csv(plasticc_data_dir+data_fname)
                X, obj_ids = create_interpolated_vectors_plasticc(data, 128)

                chunk_metadata = metadata[metadata.object_id.isin(obj_ids)]
                tags = chunk_metadata[['object_id', 'true_target']]
                tags.loc[:,'true_target'] = [plasticc_classes.index(tag) for tag in chunk_metadata["true_target"]]
        
                Y = tags.true_target.values
        
                print(obj_ids)
                print(tags)
                print(Y)

                dataset = {'X':X, 'ids':obj_ids, 'Y':Y}
                output_fname = plasticc_data_dir+'interpolated/test/plasticc_test_data_batch{}_balanced.h5'.format(i)
                save_vectors(dataset, output_fname)
                # break


# preprocess_train_data()
preprocess_test_data()


# interpolated_dataset_filename=fpath+'plasticc_test_data_batch1.h5'
# interpolated_dataset = LCs(lc_length, interpolated_dataset_filename)
# dataset_length = len(interpolated_dataset)
# print(dataset_length)
# print(interpolated_dataset[100])


