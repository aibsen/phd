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

#this experiment is for data simulation only. It has 4 parts:
#part 1: training/testing with complete light curves for all 4 models.
#part 2 : training/testing with half light curves for all 4 models, using the 1st half
#part 3 : training/testing with half light curves for all 4 models, random chunks
#part 4 : training using complete light curves and randomly cropped light curves (padded), 
#testing using padded light curves chopped randomly.
#for all of these, lc length = 128 and training is unbalanced


results_dir = "../../results/"
train_filename = "../../data/plasticc/interpolated/training/plasstic_train_data.h5"
train_filename = "../../data/plasticc/interpolated/test/plasstic_test_data_batch"

lc_length = 128
num_epochs = 100
seed = 1772670
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64


