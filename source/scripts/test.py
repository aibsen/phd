import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_utils import cached_dataset_random_split
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
cached_dataset = CachedLCs(lc_length, plasticc_dataset, data_cache_size=cache_size)
# print(len(cached_dataset))
# 2974714

batch_size = 64
num_epochs = 30
use_gpu = True
lr = 1e-03
wdc = 1e-03
n_seeds = 5


# sampler = CachedRandomSampler(cached_dataset, chunk_size=cache_size)
# sampler = RandomSampler(cached_dataset)

#split dataset into what we will use first
train_data_set_length = 125000 #so if k=5, training set is size 1000
test_data_set_length = 100000
train_dataset, test_dataset = cached_dataset_random_split(cached_dataset,cache_size,[train_data_set_length,test_data_set_length])

train_sampler= CachedRandomSampler(train_dataset, chunk_size=cache_size)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)


for idx, (x, y,ids) in enumerate(train_loader):
    print(idx)
# train_loader = torch.utils.data.DataLoader(cached_dataset, batch_size=batch_size, sampler=sampler)


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


# exp_params={
#     "num_epochs" : num_epochs,
#     "learning_rate" : lr,
#     "num_output_classes": 6,
#     "weight_decay_coefficient": wdc,
#     "use_gpu" : use_gpu,
#     "batch_size" : batch_size,
#     "sampler" : sampler
# }



# #FCN
# exp_name = "test"
# fcn = FCNN1D(fcn_params)
# exp_params["network_model"] = fcn
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
