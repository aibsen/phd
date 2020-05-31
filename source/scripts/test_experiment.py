import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import Interpolated_LCs, RandomCrop
from recurrent_models import GRU1D
from experiment import Experiment
from plot_utils import *

results_dir = "../../results/"
exp_name = "unbalanced_dataset_m_realzp_328_randomcrop111"
interpolated_dataset_filename = "../../data/training/linearly_interpolated/unbalanced_dataset_m_realzp_328.h5"
test_dataset_filename = "../../data/testing/real_data_75%p_111.h5"

lc_length = 328
lc_transform_length = 111
random_crop=RandomCrop(lc_transform_length,lc_length)


num_epochs = 30
seed = 1772670
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 1e-03

#load dataset
interpolated_dataset = Interpolated_LCs(lc_length, interpolated_dataset_filename, transform=random_crop)
dataset_length = len(interpolated_dataset)

#split into train/validation, validation will be ~ .3
val_length = int(dataset_length/4)
# test_length = int(dataset_length/4)
# train_length = dataset_length - val_length -test_length
train_length = dataset_length - val_length
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length, test_length])
train_dataset, val_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length])
#
test_dataset = Interpolated_LCs(lc_length, test_dataset_filename)
test_length = len(test_dataset)

#dataset loaders
print("training set length: ",str(train_length))
print("validation set length: ",str(val_length))
print("test set length: ",str(test_length))

batch_size = 64

#this bit is to balance dataset
# weights = [0.25,0.25,0.25,0.25]
# num_samples = 50000
# train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

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
experiment.run_experiment(test_results="test_results_75%p.csv",test_summary="test_summary_75%p.csv")
print("--- %s seconds ---" % (time.time() - start_time))

validation_results = pd.read_csv(results_dir+exp_name+"/result_outputs/validation_results.csv")
true_tagsv = validation_results.true_tags.values
predictedv = validation_results.predicted_tags.values
plot_cm(true_tagsv,predictedv)
plot_cm(true_tagsv,predictedv,False)


test_results = pd.read_csv(results_dir+exp_name+"/result_outputs/test_results_75%p.csv")
true_tags = test_results.true_tags.values
predicted = test_results.predicted_tags.values
plot_cm(true_tags,predicted)
plot_cm(true_tags,predicted,False)
