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
cache_size = 300000
cached_dataset = CachedLCs(lc_length, plasticc_dataset, chunk_size=cache_size)

batch_size = 64
use_gpu = True
seeds = [782915]
torch.manual_seed(seed=seeds[0])


# ############ PART 1 ###############
# testing only, repeatedly for different test sets.
# #testing using padded simulated light curves chopped. 
# 10%-25,50,75 and 100 of lcs

training_size = ["10e+3","5x10e+3","10e+4","5x10e+4","10e+5","5x10e+5"]
percents = [0.25,0.5,0.75]
percents_str = ["25p","50p","75p"]
test_data_set_length = 100000
test_dataset = cached_dataset_random_split(cached_dataset,[test_data_set_length],cache_size)[0]
input_shape = test_dataset[0][0].shape

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
    "num_output_classes": 6,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "chunk_size": cache_size
}
for ts in training_size:
    for p,pstr in zip(percents,percents_str):

        results_file="test_results_{}.csv".format(pstr) 
        summary_file="test_summary_{}.csv".format(pstr)

        transform = transforms.Compose([RandomCrop(int(lc_length*p),lc_length), ZeroPad(lc_length, int(lc_length*p))])
        test_dataset.transform = transform

        # RNN
        exp_name = "exp_plasticc_gru_{}".format(ts)
        gru = GRU1D(gru_params)
        exp_params["network_model"] = gru
        experiment = SeededExperiment(
            results_dir+exp_name,
            exp_params,
            test_data=test_dataset,
            verbose=True)

        seeds = experiment.get_seeds_from_folders()
        experiment.seeds = seeds
        start_time = time.time()
        experiment.run_experiment(test_results=results_file, test_summary=summary_file)
        print("--- %s seconds ---" % (time.time() - start_time))


        #RNN-attention
        exp_name = "exp_plasticc_grusa_{}".format(ts)
        grusa = GRU1D(grusa_params)
        exp_params["network_model"] = grusa
        experiment = SeededExperiment(
            results_dir+exp_name,
            exp_params,
            test_data=test_dataset,
            verbose=True)

        seeds = experiment.get_seeds_from_folders()
        experiment.seeds = seeds

        start_time = time.time()
        experiment.run_experiment(test_results=results_file, test_summary=summary_file)
        print("--- %s seconds ---" % (time.time() - start_time))

        #FCN
        exp_name = "exp_plasticc_fcn_{}".format(ts)
        fcn = FCNN1D(fcn_params)
        exp_params["network_model"] = fcn
        experiment = SeededExperiment(
            results_dir+exp_name,
            exp_params,
            test_data=test_dataset,
            verbose=True)

        seeds = experiment.get_seeds_from_folders()
        experiment.seeds = seeds
        start_time = time.time()
        experiment.run_experiment(test_results=results_file, test_summary=summary_file)
        print("--- %s seconds ---" % (time.time() - start_time))

        #ResNet
        exp_name = "exp_plasticc_resnet_{}".format(ts)
        resnet = ResNet1D(resnet_params)
        exp_params["network_model"] = resnet
        experiment = SeededExperiment(
            results_dir+exp_name,
            exp_params,
            test_data=test_dataset,
            verbose=True)

        seeds = experiment.get_seeds_from_folders()
        experiment.seeds = seeds

        start_time = time.time()
        experiment.run_experiment(test_results=results_file, test_summary=summary_file)

        print("--- %s seconds ---" % (time.time() - start_time))

