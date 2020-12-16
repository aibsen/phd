import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs, CachedLCs 
from transforms import RandomCrop,ZeroPad,RightCrop
from recurrent_models import GRU1D
from convolutional_models import FCNN1D, ResNet1D
from experiment import Experiment
from plot_utils import *
from torchvision import transforms

results_dir = "../../results/"
interpolated_dataset_filename = "../../data/training/linearly_interpolated/unbalanced_dataset_m_realzp_128_3types.h5"

lc_length = 128
num_epochs = 100
seed = 1772670
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64


fcn_params = {
    "num_output_classes" : 3,
    "regularize" : False,
    "global_pool" : 'max'
}
resnet_params = {
    "num_output_classes" : 3,
    "global_pool":'avg',
    "n_blocks":3
}
gru_params = {
    "num_output_classes" : 3,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"no_attention",
    "da":50,
    "r":1
    }
grusa_params = {
    "num_output_classes" :3,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"self_attention",
    "da":50,
    "r":1
    }
params = [fcn_params, resnet_params, gru_params, grusa_params]

# test_dataset_filenames = ["../../data/testing/real_data_25%p_15.h5",
#                         "../../data/testing/real_data_50%p_22.h5",
#                         "../../data/testing/real_data_75%p_44.h5"]
# test_lc_lengths = [15,22,44]
# test_percentiles = [25,50,75]
# lc_lens = [128,64,64,128,32,32,128,128,128,128,128] #length input shape of trained classifiers
results_dir = "../../results/"

# def load_dataset(lc_length, test_dataset_filename, transform=None):
#     test_dataset = LCs(lc_length, test_dataset_filename, transform=transform)
#     test_length = len(test_dataset)
#     print("test set length: ",str(test_length))
#     print(test_dataset[0][0].shape)
#     test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)
#     return test_loader, test_dataset[0][0].shape

def load_datasets(lc_length, interpolated_dataset_filename, transform=None):
    #load dataset
    interpolated_dataset = LCs(lc_length, interpolated_dataset_filename, transform=transform)
    dataset_length = len(interpolated_dataset)

    #split into train/validation/test, validation/test will be ~ .4
    val_length = int(dataset_length/4)
    test_length = int(dataset_length/4)
    train_length = dataset_length - val_length -test_length
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length, test_length])

    #dataset loaders
    print("training set length: ",str(train_length))
    print("validation set length: ",str(val_length))
    print("test set length: ",str(test_length))

    batch_size = 64

    return train_dataset, val_dataset, test_dataset, train_dataset[0][0].shape

def find_best_epoch(f):
    results_summary = pd.read_csv(f)
    val_f1 = results_summary.val_f1.values
    best_epoch = val_f1.argmax()
    return best_epoch

# for i in np.arange(7,12): #for each experiment

# exp_names = list(map(lambda x: x.format(i),["simonly_p{}_fcn", "simonly_p{}_resnet", "simonly_p{}_gru", "simonly_p{}_grusa"]))
    # for l,test_dataset_filename in list(zip(test_lc_lengths,test_dataset_filenames)): #for each test set

exp_names = ["simonly_p1_fcn", "simonly_p1_resnet", "simonly_p1_gru", "simonly_p1_grusa"]
train_dataset, val_dataset, test_dataset, input_shape = load_datasets(lc_length,interpolated_dataset_filename)


for m,param in enumerate(params): #for each model in the experiment 
    best_epoch = find_best_epoch(results_dir+exp_names[m]+"/result_outputs/summary.csv")

        # for j,test_dataset_filename in enumerate(test_dataset_filenames): #classify each test set with each model in each exp
            # zeropad = ZeroPad(lc_lens[i-1],test_lc_lengths[j])
            # test_loader, input_shape = load_dataset(lc_lens[i-1],test_dataset_filename,zeropad)
            
    param["input_shape"] = input_shape

    if m == 0: #fcn
        network = FCNN1D(param)
    elif m == 1: #resnet
        network = ResNet1D(param)
    elif m == 2: #gru
        network = GRU1D(param)
    elif m == 3: #grusa
        network = GRU1D(param)


    experiment = Experiment(
        network_model = network,
        experiment_name = results_dir+exp_names[m],
        num_epochs = num_epochs,
        learning_rate = lr,
        weight_decay_coefficient = wdc,
        use_gpu = use_gpu,
        test_data = test_dataset,
        best_idx = best_epoch
    )
    start_time = time.time()
    experiment.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))

    