import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
from recurrent_models import GRU1D
from convolutional_models import FCNN1D, ResNet1D
from seeded_experiment import SeededExperiment
from plot_utils import *
from torchvision import transforms
from transforms import RandomCrop,ZeroPad,RightCrop

results_dir = "../../results/"
# real_datasets = [
#     "../../data/testing/real_data_count5_careful.h5",
#     "../../data/testing/real_data_count10_careful.h5",
#     "../../data/testing/real_data_count15_careful.h5",
#     "../../data/testing/real_data_count20_careful.h5",
#     "../../data/testing/real_data_count25_careful.h5",
#     "../../data/testing/real_data_count30_careful.h5",
#     "../../data/testing/real_data_count35_careful.h5",
#     "../../data/testing/real_data_count40_careful.h5"]

lc_length = 128
num_epochs = 30
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64
n_seeds = 5

############ PART 1 ###############
#testing using padded real light curves chopped. 10% of lcs different test sets

counts = [5,10,15,20,25,30,35,40]
for count in counts: #interate over different test sets
    c = str(count)
    test_data_file = "../../data/testing/real_data_count{}_careful.h5".format(c)
    test_results = "test_results_count{}.csv".format(c)
    test_summary = "test_results_summary{}.csv".format(c)
    
    #load dataset
    test_dataset = LCs(lc_length, test_data_file)
    test_length = len(test_dataset)
    input_shape = test_dataset[0][0].shape

    #define network params
    fcn_params = {
        "input_shape": input_shape,
        "num_output_classes" : 4,
        "regularize" : False,
        "global_pool" : 'max'
    }
    resnet_params = {
        "input_shape": input_shape,
        "num_output_classes" : 4,
        "global_pool":'avg',
        "n_blocks":3
    }
    gru_params = {
        "input_shape": input_shape,
        "num_output_classes" : 4,
        "hidden_size":100,
        "batch_size":batch_size,
        "attention":"no_attention",
        "da":20,
        "r":3
        }
    grusa_params = {
        "input_shape": input_shape,
        "num_output_classes" : 4,
        "hidden_size":100,
        "batch_size":batch_size,
        "attention":"self_attention",
        "da":50,
        "r":1
        }

    exp_params={
        "num_epochs" : num_epochs,
        "learning_rate" : lr,
        "weight_decay_coefficient" : wdc,
        "use_gpu" : use_gpu,
        "batch_size" : batch_size
    }

    #2.C RNN
    exp_name = "exp2_p1_gru"
    gru = GRU1D(gru_params)
    exp_params["network_model"] = gru
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        test_data=test_dataset,
        verbose=True,
        n_seeds=n_seeds)
    
    seeds = experiment.get_seeds_from_folders()
    experiment.seeds = seeds

    start_time = time.time()
    experiment.run_experiment(test_results,test_summary)
    print("--- %s seconds ---" % (time.time() - start_time))


    #2.D RNN-attention
    exp_name = "exp2_p1_grusa"
    grusa = GRU1D(grusa_params)
    exp_params["network_model"] = grusa
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        test_data=test_dataset,
        verbose=True,
        n_seeds=n_seeds)

    seeds = experiment.get_seeds_from_folders()
    experiment.seeds = seeds
    start_time = time.time()
    experiment.run_experiment(test_results,test_summary)
    print("--- %s seconds ---" % (time.time() - start_time))

    #2.A FCN
    exp_name = "exp2_p1_fcn"
    fcn = FCNN1D(fcn_params)
    exp_params["network_model"] = fcn
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        test_data=test_dataset,
        verbose=True,
        n_seeds=n_seeds)

    seeds = experiment.get_seeds_from_folders()
    experiment.seeds = seeds
    start_time = time.time()
    experiment.run_experiment(test_results,test_summary)
    print("--- %s seconds ---" % (time.time() - start_time))

    #2.B ResNet
    exp_name = "exp2_p1_resnet"
    resnet = ResNet1D(resnet_params)
    exp_params["network_model"] = resnet
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        test_data=test_dataset,
        verbose=True,
        n_seeds=n_seeds)

    seeds = experiment.get_seeds_from_folders()
    experiment.seeds = seeds
    start_time = time.time()
    experiment.run_experiment(test_results,test_summary)
    print("--- %s seconds ---" % (time.time() - start_time))

