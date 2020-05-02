import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import sys  
from datasets import Interpolated_LCs
from recurrent_models import GRU1D
from experiment import Experiment

results_dir = "../results/"
exp_name = "test_experiment"
num_epochs = 100

lc_length = 128
seed = 1772670
torch.manual_seed(seed=seed) 
use_gpu = True
lr = 1e-03
wdc = 0

#load dataset
interpolated_dataset_filename = "../data/training/linearly_interpolated/sn_0000000_m.h5"
interpolated_dataset = Interpolated_LCs(lc_length, interpolated_dataset_filename)
dataset_length = len(interpolated_dataset)

#split into train/validation, validation will be ~ .3
val_length = int(dataset_length/3)
train_length = dataset_length - val_length
train_dataset, val_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length])

test_dataset_filename = "../data/testing/test_40.h5"
test_dataset = Interpolated_LCs(lc_length, test_dataset_filename)
test_length = len(test_dataset)

#dataset loaders
print("training set length: ",str(train_length))
print("validation set length: ",str(val_length))
print("test set length: ",str(test_length))

batch_size = 64

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_length, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=test_length,shuffle=True)

#define network parameters
params = {
    "input_shape": train_dataset[0][0].shape,
    "num_output_classes" : 4,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"self_attention",
    "da":50,
    "r":1
    }

gru_self_attention = GRU1D(params)

experiment = Experiment(
    network_model = gru_self_attention,
    experiment_name = results_dir+exp_name,
    num_epochs = num_epochs,
    learning_rate = lr,
    weight_decay_coefficient = wdc,
    use_gpu = use_gpu,
    train_data = train_loader,
    val_data = val_loader,
    test_data = test_loader
)
start_time = time.time()
experiment.run_experiment()
print("--- %s seconds ---" % (time.time() - start_time))