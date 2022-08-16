from fileinput import filename
import h5py
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_data_utils import * 


data_dir0 = '../../data/plasticc/interpolated/training/'
data_dir = '../../data/plasticc/interpolated/test/'
input_filename = 'plasticc_test_data_batch'
output_filename = input_filename+'_extragalactic'

def filter_extragalactic_objects(input_filename, output_filename):
    try:
        X = None
        Y = None
        id = None

        with h5py.File(data_dir+input_filename+'.h5','r') as f:
            Y = f["Y"]
            Y = torch.tensor(Y, dtype=torch.long)
            extra_galactic_idx = torch.where(Y<9)[0]
            Y = Y[extra_galactic_idx]

            ids = f["ids"]
            ids = torch.tensor(ids,dtype=torch.long)
            ids = ids[extra_galactic_idx]
            
            X = f["X"][:,0:12]
            X = torch.tensor(X,dtype=torch.float)
            X = X[extra_galactic_idx]

        print(Y.shape)
        print(ids.shape)
        print(X.shape)

        dataset = {
            'X': X,
            'Y': Y,
            'ids': ids
        }

        save_vectors(dataset, data_dir+output_filename+'.h5')

    except Exception as e:
        print(e)
def filter_extragalactic_objects_test():
    for i in range(3,12):
        # input_fn = input_filename+str(i)
        input_fn = 'plasticc_test_data_batch{}_balanced'.format(i)
        output_fn = input_fn+'_extragalactic'
        filter_extragalactic_objects(input_fn, output_fn)
