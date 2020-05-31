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
exp_name = "unbalanced_dataset_m_realzp_328_110"
num_epochs = 30

lc_length = 128
seed = 1772670
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 1e-03

#load dataset
interpolated_dataset_filename = "../../data/training/linearly_interpolated/unbalanced_dataset_m_realzp_128.h5"
random_crop=RandomCrop(54,128)
interpolated_dataset = Interpolated_LCs(lc_length, interpolated_dataset_filename,transform=random_crop)
dataset_length = len(interpolated_dataset)

#split into train/validation, validation will be ~ .4
val_length = int(dataset_length/4)
# test_length = int(dataset_length/4)
# train_length = dataset_length - val_length -test_length
train_length = dataset_length - val_length
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length, test_length])
train_dataset, val_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length])
#
train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


for idx, (x, y,ids) in enumerate(train_data):
    print(x,y,ids)
    print(x.shape)
    break