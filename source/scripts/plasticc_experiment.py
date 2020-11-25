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
cache_size = 200000
cached_dataset = CachedLCs(lc_length, plasticc_dataset, chunk_size=cache_size)

batch_size = 64
num_epochs = 30
use_gpu = True
lr = 1e-03
wdc = 1e-03
n_seeds = 1
seeds = [782915]
torch.manual_seed(seed=seeds[0])


# ############ PART 1 ###############
# #training using complete simulated light curves and cropped light curves (1st bits) (50% + 25%+10%) (padded) 
# #testing using padded simulated light curves chopped. 10% of lcs
# #training set size 10e+3,5x10e+3,10e+4,5x10e+4,10e+5,5x10e+5, cached LCs still not needed 
# "10e+3","5x10e+3","10e+4","5x10e+4","10e+5",
training_sizes_str= ["5x10e+5"]
# 1250,6250,12500,62500,125000,
training_sizes= [625000]
#split dataset into what we will use first
test_data_set_length = 100000

for ts, ts_str in zip(training_sizes, training_sizes_str):
    train_dataset, test_dataset = cached_dataset_random_split(cached_dataset,[ts,test_data_set_length],cache_size)


    #crop training dataset into different lengths
    lengths = np.array([0.1, 0.25, 0.5, 1.0])*lc_length
    lengths = [int(l) for l in lengths]
    transform = RandomCropsZeroPad(lengths, lc_length)
    train_dataset.transform = transform

    #crop test dataset so it's 10% of lcs
    transform2 = transforms.Compose([RandomCrop(int(lc_length*0.1),lc_length), ZeroPad(lc_length, int(lc_length*0.1))])
    test_dataset.transform = transform
    input_shape = train_dataset[0][0].shape

    #define network params
    fcn_params = {
        "input_shape": input_shape,
        "num_output_classes" : 6,
        "regularize" : False,
        "global_pool" : 'max'
    }
    resnet_params = {
        "input_shape": input_shape,
        "num_output_classes" : 6,
        "global_pool":'avg',
        "n_blocks":3
    }
    gru_params = {
        "input_shape": input_shape,
        "num_output_classes" : 6,
        "hidden_size":100,
        "batch_size":batch_size,
        "attention":"no_attention",
        "da":50,
        "r":1
        }
    grusa_params = {
        "input_shape": input_shape,
        "num_output_classes" : 6,
        "hidden_size":100,
        "batch_size":batch_size,
        "attention":"self_attention",
        "da":50,
        "r":1
        }

    exp_params={
        "num_epochs" : num_epochs,
        "learning_rate" : lr,
        "num_output_classes": 6,
        "weight_decay_coefficient": wdc,
        "use_gpu" : use_gpu,
        "batch_size" : batch_size,
        "chunk_size": cache_size
    }


    # RNN
    exp_name = "exp_plastic_gru_{}".format(ts_str)
    gru = GRU1D(gru_params)
    exp_params["network_model"] = gru
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        train_data=train_dataset,
        test_data=test_dataset,
        verbose=True,
        seeds = seeds,
        n_seeds=n_seeds)

    start_time = time.time()
    experiment.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))


    #RNN-attention
    exp_name = "exp_plasticc_grusa_{}".format(ts_str)
    gru = GRU1D(gru_params)
    grusa = GRU1D(grusa_params)
    exp_params["network_model"] = grusa
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        train_data=train_dataset,
        test_data=test_dataset,
        verbose=True,
        seeds = seeds,
        n_seeds=n_seeds)

    start_time = time.time()
    experiment.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))

    #FCN
    exp_name = "exp_plasticc_fcn_{}".format(ts_str)
    gru = GRU1D(gru_params)
    fcn = FCNN1D(fcn_params)
    exp_params["network_model"] = fcn
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        train_data=train_dataset,
        test_data=test_dataset,
        verbose=True,
        seeds = seeds,
        n_seeds=n_seeds)
    start_time = time.time()
    experiment.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))

    #ResNet
    exp_name = "exp_plasticc_resnet_{}".format(ts_str)
    gru = GRU1D(gru_params)
    resnet = ResNet1D(resnet_params)
    exp_params["network_model"] = resnet
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        train_data=train_dataset,
        test_data=test_dataset,
        verbose=True,
        seeds = seeds,
        n_seeds=n_seeds)

    start_time = time.time()
    experiment.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))

