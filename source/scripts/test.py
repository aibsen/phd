import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_utils import *
from datasets import CachedLCs, LCs
from data_samplers import CachedRandomSampler
from recurrent_models import GRU1D
from experiment import Experiment
from plot_utils import *
from torchvision import transforms
from transforms import RandomCrop,ZeroPad,RightCrop,RandomCropsZeroPad
from recurrent_models import GRU1D
from convolutional_models import FCNN1D, ResNet1D
from seeded_experiment import SeededExperiment

from torch.utils.data import RandomSampler

results_dir = "../../results/"
exp_name = "plasticc_exp"
plasticc_dataset = "../../data/plasticc/plasticc_dataset.h5"
lc_length = 128
cache_size = 100000
cached_dataset = CachedLCs(lc_length, plasticc_dataset, chunk_size=cache_size)

batch_size = 64
num_epochs = 1
use_gpu = True
lr = 1e-03
wdc = 1e-03
n_seeds = 1

#split dataset into what we will use first
train_data_set_length = 125000 #so if k=5, training set is size 1000
test_data_set_length = 100000
train_dataset, test_dataset = cached_dataset_random_split(cached_dataset,[train_data_set_length,test_data_set_length],cache_size)
print(train_dataset)
print(train_dataset.indices)
print(len(train_dataset.indices))
print(test_dataset)
print(test_dataset.indices)
print(len(test_dataset.indices))
print("")


#crop training dataset into different lengths
lengths = np.array([0.1, 0.25, 0.5, 1.0])*lc_length
lengths = [int(l) for l in lengths]
transform = RandomCropsZeroPad(lengths, lc_length)
train_dataset.transform = transform

k=5
train_length = len(train_dataset)
kf_length = int(train_length/k)
print(kf_length)
kf_lengths = [kf_length]*k
print(kf_lengths)
kfs = cached_crossvalidator_split(train_dataset,kf_lengths,cache_size)
print("")
for j,(tr,val) in enumerate(kfs):
    print("K = "+str(j))
    # print("")
    # print("tr")
    # print(tr)
    # print("val")
    # print(val)
    train_indices = train_dataset.indices[tr]
    print(train_indices)
    val_indices = train_dataset.indices[val]
    print(val_indices)
    train_dataset_i = CachedLCs(train_dataset.lc_length, train_dataset.dataset_file,cache_size,len(train_indices),train_indices,train_dataset.transform)
    val_dataset_i = CachedLCs(train_dataset.lc_length, train_dataset.dataset_file,cache_size,len(val_indices),val_indices,train_dataset.transform)
    
    train_sampler = CachedRandomSampler(train_dataset_i)
    val_sampler = CachedRandomSampler(val_dataset_i)



    train_loader = torch.utils.data.DataLoader(train_dataset_i, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset_i,batch_size=batch_size,sampler=val_sampler)

    print("len tain loader:"+str(len(train_loader)))

    for idx, (x, y,ids) in enumerate(train_loader):
        print("")

    print("len val loader:"+str(len(val_loader)))
    for x, y,ids in val_loader:
        print("")

            
# #crop training dataset into different lengths
# lengths = np.array([0.1, 0.25, 0.5, 1.0])*lc_length
# lengths = [int(l) for l in lengths]
# transform = RandomCropsZeroPad(lengths, lc_length)
# train_dataset.transform = transform

# #crop test dataset so it's 10% of lcs
# transform2 = transforms.Compose([RandomCrop(int(lc_length*0.1),lc_length), ZeroPad(lc_length, int(lc_length*0.1))])

# input_shape = train_dataset[0][0].shape

# #define network params
# fcn_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 6,
#     "regularize" : False,
#     "global_pool" : 'max'
# }
# resnet_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 6,
#     "global_pool":'avg',
#     "n_blocks":3
# }
# gru_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 6,
#     "hidden_size":100,
#     "batch_size":batch_size,
#     "attention":"no_attention",
#     "da":50,
#     "r":1
#     }
# grusa_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 6,
#     "hidden_size":100,
#     "batch_size":batch_size,
#     "attention":"self_attention",
#     "da":50,
#     "r":1
#     }

# exp_params={
#     "num_epochs" : num_epochs,
#     "learning_rate" : lr,
#     "num_output_classes": 6,
#     "weight_decay_coefficient": wdc,
#     "use_gpu" : use_gpu,
#     "batch_size" : batch_size,
#     "chunk_size": cache_size
# }



# # RNN
# exp_name = "exp_plastic_gru_5x10+5"
# gru = GRU1D(gru_params)
# exp_params["network_model"] = gru
# experiment = SeededExperiment(
#     results_dir+exp_name,
#     exp_params,
#     train_data=train_dataset,
#     test_data=test_dataset,
#     verbose=True,
#     n_seeds=n_seeds)

# start_time = time.time()
# experiment.run_experiment()
# print("--- %s seconds ---" % (time.time() - start_time))

