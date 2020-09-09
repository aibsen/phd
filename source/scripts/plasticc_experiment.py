import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import CachedLCs
from data_samplers import CachedRandomSampler
from recurrent_models import GRU1D
from experiment import Experiment
from plot_utils import *

results_dir = "../../results/"
exp_name = "plasticc_exp"
plasticc_dataset = "../../data/plasticc/plasticc_dataset.h5"
lc_length = 128
cache_size = 500000
cached_dataset = CachedLCs(lc_length, plasticc_dataset, data_cache_size=cache_size)
batch_size = 64


sampler = CachedRandomSampler(cached_dataset, chunk_size=cache_size)
train_loader = torch.utils.data.DataLoader(cached_dataset, batch_size=batch_size, sampler=sampler)

for i,(x,y,ido) in enumerate(train_loader):
    print(i)
    # print(ido)
    # break
    