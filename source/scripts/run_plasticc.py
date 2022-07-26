from sqlite3 import paramstyle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import  LCs
from seeded_experiment import SeededExperiment
from transforms import GroupClass
from convolutional_models import FCNN1D, ResNet1D

results_dir = "../../results/"
data_dir = "../../data/plasticc/interpolated/"
exp_name = results_dir+"plasticc_vanilla_fcn"


lc_length = 128
batch_size = 64
num_epochs = 10
use_gpu = True
lr = 1e-03
wdc = 1e-03
seeds = [1772670]
seed=torch.cuda.manual_seed(seed=1772670)
n_seeds = 1


####Vanilla Plasticc
training_data_file=data_dir+'training/plasticc_train_data.h5'
train_dataset = LCs(lc_length, training_data_file)
input_shape = train_dataset[0][0].shape

test_data_file = data_dir+'test/plasticc_test_data_batch1.h5'
transform = GroupClass(14,14)
test_dataset = LCs(lc_length, test_data_file, transform=transform)

fcn_params = {
    "input_shape" : input_shape,
    "num_output_classes":15    
}


network = FCNN1D(fcn_params)

exp_params={
    "network_model": network,
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "num_output_classes": 15,
    "patience":5,
    "validation_step":3
}

experiment = SeededExperiment( 
    exp_name,
    exp_params = exp_params,
    seeds = [1772670, 123],
    train_data=train_dataset,
    test_data=test_dataset
    )

experiment.run_experiment()



    # if m == 0: #fcn
    #     network = FCNN1D(param)
    #     exp_n = exp_name+"_fcn"

    # elif m == 1: #resnet
    #     network = ResNet1D(param)
    #     exp_n = exp_name+"_resnet"

    # elif m == 2: #gru
    #     network = GRU1D(param)
    #     exp_n = exp_name+"_gru"

    # elif m == 3: #grusa
    #     network = GRU1D(param)
    #     exp_n = exp_name+"_grusa"

    # #train phase
    # exp_params["network_model"] = network
    # experiment = SeededExperiment(
    #     results_dir+exp_n,
    #     exp_params,
    #     train_data=train_dataset,
    #     verbose=True,
    #     seeds = seeds)
    # start_time = time.time()
    # experiment.run_experiment()
    # print("--- %s seconds in train phase---" % (time.time() - start_time))

    # experiment.train_data = None #there is probably a more elegant way to do this, but we will leave it for now

    # #test_phase
    # for i in np.arange(1,12):
    #     test_data_file=data_dir+'plasticc_test_data_batch{}.h5'.format(i)
    #     test_dataset = LCs(lc_length, test_data_file)
    #     experiment.test_data = test_dataset

    #     test_results = 'test_results{}.csv'.format(i)
    #     test_summary = 'test_summary{}.csv'.format(i)
    #     start_time = time.time()
    #     experiment.run_experiment(test_results,test_summary)
    #     print("--- %s seconds ---" % (time.time() - start_time))







