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
interpolated_dataset_filename = "../../data/training/linearly_interpolated/rapid_data.h5"
real_dataset_filename = "../../data/testing/real_data_30_careful.h5"

lc_length = 128
num_epochs = 30
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64
n_seeds = 5

############ PART 1 ###############
#training using complete simulated light curves and cropped light curves (1st bits) (50% + 25%+10%) (padded) 
#testing using padded real light curves chopped. 10% of lcs

#load dataset
train_dataset = LCs(lc_length, interpolated_dataset_filename)
train_length = len(train_dataset)
test_dataset = LCs(lc_length, real_dataset_filename)
test_length = len(test_dataset)


#apply transforms so we'll be training with 100, 50, 25 and 10 percent of light curves
"""I need to change this because I changed my definition of earliness"""
if train_dataset[0][0].shape == test_dataset[0][0].shape:
    train_length1 = int(train_length/4)
    train_length2 = int(train_length/4)
    train_length3 = int(train_length/4)
    train_length4 = train_length - train_length1 - train_length2 - train_length3
    trd1, trd2, trd3, trd4  = torch.utils.data.random_split(train_dataset, [train_length1,train_length2, train_length3, train_length4])


    for p,trd in list(zip([0.1,0.25,0.5],[trd1,trd2,trd3])):
        crop=RandomCrop(int(lc_length*p),lc_length)
        zeropad = ZeroPad(lc_length,int(lc_length*p))
        composed = transforms.Compose([crop,zeropad])
        trd.transform = composed

    train_dataset = torch.utils.data.ConcatDataset([trd1,trd2,trd3,trd4])

    input_shape = train_dataset[0][0].shape

    #define network params
    fcn_params = {
        "input_shape": input_shape,
        "num_output_classes" : 11,
        "regularize" : False,
        "global_pool" : 'max'
    }
    resnet_params = {
        "input_shape": input_shape,
        "num_output_classes" : 11,
        "global_pool":'avg',
        "n_blocks":11
    }
    gru_params = {
        "input_shape": input_shape,
        "num_output_classes" : 11,
        "hidden_size":100,
        "batch_size":batch_size,
        "attention":"no_attention",
        "da":50,
        "r":1
        }
    grusa_params = {
        "input_shape": input_shape,
        "num_output_classes" : 11,
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
        "batch_size" : batch_size,
        "chunked": False,
        "num_output_classes": 11
    }
    #2.C RNN
    exp_name = "exp2_rapid_p1_gru"
    gru = GRU1D(gru_params)
    exp_params["network_model"] = gru
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        train_data=train_dataset,
        test_data=test_dataset,
        verbose=True,
        n_seeds=n_seeds)

    start_time = time.time()
    experiment.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))


    #2.D RNN-attention
    exp_name = "exp2_rapid_p1_grusa"
    grusa = GRU1D(grusa_params)
    exp_params["network_model"] = grusa
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        train_data=train_dataset,
        test_data=test_dataset,
        verbose=True,
        n_seeds=n_seeds)

    start_time = time.time()
    experiment.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))

    #2.A FCN
    exp_name = "exp2_rapid_p1_fcn"
    fcn = FCNN1D(fcn_params)
    exp_params["network_model"] = fcn
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        train_data=train_dataset,
        test_data=test_dataset,
        verbose=True,
        n_seeds=n_seeds)
    start_time = time.time()
    experiment.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))

    #2.B ResNet
    exp_name = "exp2_rapid_p1_resnet"
    resnet = ResNet1D(resnet_params)
    exp_params["network_model"] = resnet
    experiment = SeededExperiment(
        results_dir+exp_name,
        exp_params,
        train_data=train_dataset,
        test_data=test_dataset,
        verbose=True,
        n_seeds=n_seeds)

    start_time = time.time()
    experiment.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))



else:
    print("training and test set need to be the same length")
