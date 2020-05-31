import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import Interpolated_LCs
from recurrent_models import GRU1D
from experiment import Experiment
from plot_utils import *

results_dir = "../../results/"
exp_name = "unbalanced_dataset_m_realzp_128"
num_epochs = 100

lc_length = 15
seed = 1772670
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 0

test_dataset_filename = "../../data/testing/real_data_25%p_15.h5"
test_dataset = Interpolated_LCs(lc_length, test_dataset_filename)
test_length = len(test_dataset)

#dataset loaders
print("test set length: ",str(test_length))

batch_size = 64

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)

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
    best_idx=1
)
start_time = time.time()
experiment.run_test_phase(test_loader,"test_results25%p_15.csv","test_results25%p_15_summary.csv")

print("--- %s seconds ---" % (time.time() - start_time))

test_results = pd.read_csv(results_dir+exp_name+"/result_outputs/test_results25%p_15.csv")
true_tags = test_results.true_tags.values
predicted = test_results.predicted_tags.values
plot_cm(true_tags,predicted)
plot_cm(true_tags,predicted,False)
