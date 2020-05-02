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

test_dataset_filename = "../data/testing/test_40.h5"
test_dataset = Interpolated_LCs(lc_length, test_dataset_filename)
test_length = len(test_dataset)

#dataset loaders
print("test set length: ",str(test_length))

batch_size = 64

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=test_length,shuffle=True)

#define network parameters
params = {
    "input_shape": test_dataset[0][0].shape,
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
    test_data = test_loader,
    continue_from_epoch=11
)
start_time = time.time()
experiment.run_test_phase(test_loader,"test_results2.csv","test_summary2.csv")

print("--- %s seconds ---" % (time.time() - start_time))