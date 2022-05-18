import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import CachedLCs, LCs
from data_samplers import CachedRandomSampler
from experiment import Experiment
# from plot_utils import *
from torchvision import transforms
from transforms import RandomCrop,ZeroPad,RightCrop,RandomCropsZeroPad
from recurrent_models import GRU1D
from convolutional_models import CNN
from dmdt_cnns import DMDTShallowCNN, DMDTCNN
from seeded_experiment import SeededExperiment
from experiment import Experiment 
import matplotlib.pyplot as plt

from torch.utils.data import RandomSampler

results_dir = "../../results/"
data_dir = "/home/ai/phd/data/plasticc/dmdt/"
exp_name = results_dir+"dmdt_24x24_1"


lc_length = 24
batch_size = 64
num_epochs = 100
use_gpu = True
lr = 1e-03
wdc = 1e-03
seeds = [1772670]
seed=torch.cuda.manual_seed(seed=1772670)
n_seeds = 1


training_data_file=data_dir+'dmdts_training_24x24.h5'
train_dataset = LCs(lc_length,training_data_file,n_channels=6)
input_shape = train_dataset[0][0].shape
print(input_shape)

####DMDT Plasticc
cnn_params={
 'input_shape': input_shape,
 'n_filters': 32,
 'kernel_size':3,
 "num_output_classes": 14,
 "n_layers":3
}

network = DMDTCNN(cnn_params)

exp_params={
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "chunked": False,
    "num_output_classes": 14,
    "network_model":network
}

experiment = SeededExperiment(
    exp_name = exp_name,
    exp_params = exp_params,
    seeds = seeds,
    train_data = train_dataset
)

experiment.run_experiment()

