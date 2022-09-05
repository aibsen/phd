from cmath import exp
from sqlite3 import paramstyle
from struct import pack
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import  LCs
from experiment import Experiment
from seeded_experiment import SeededExperiment
from transforms import CropPadTensor, GroupClass, GroupClassTensor, MultiCropPadTensor
from convolutional_models import FCNN1D, ResNet1D
from recurrent_models import GRU1D

results_dir = "../../results/"
data_dir = "../../data/plasticc/interpolated/"
# exp_name = results_dir+"plasticc_balanced_cropped_resnet"


lc_length = 128
batch_size = 64
num_epochs = 100
use_gpu = True
lr = 1e-04
wdc = 1e-02
seeds = [1772670]
seed=torch.cuda.manual_seed(seed=1772670)
n_seeds = 1
num_classes = 9
patience = 5

exp_params={
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "num_output_classes": num_classes,
    "patience":3,
    "validation_step":3
}

####Vanilla Plasticc
transform1 = CropPadTensor(lc_length, cropping=0.25)

for i in range(3,12):
    test_data_file=data_dir+'test/plasticc_test_data_batch{}_balanced_egx.h5'.format(i)
    test_dataset = LCs(lc_length, test_data_file,n_classes=num_classes, transforms=[transform1])
    test_dataset.load_data_into_memory()
    input_shape = test_dataset[0][0].shape
    test_dataset.apply_tranforms()
    # test_dataset.packed = True
    # test_dataset.lens=torch.full((len(test_dataset),),int(lc_length*0.5))

    for arch in range(0,4):

        nn_params = {
            "input_shape" : input_shape,
            "num_output_classes":num_classes,
            "hidden_size" : 100
            # "attention" : True    
        }
        
        if arch>1:
            exp_name = results_dir+"plasticc_balanced_gru_eg"
            # train_dataset.packed = True
            # val_dataset.packed = True
            test_dataset.packed = True
            # test_dataset.lens=torch.full((len(test_dataset),),int(lc_length*0.5))

            if arch == 3:
                exp_name = results_dir+"plasticc_balanced_grusa_eg"
                nn_params["attention"] = True

            network = GRU1D(nn_params)

        elif arch == 0:
            exp_name = results_dir+"plasticc_balanced_fcn_eg"
            network = FCNN1D(nn_params)
        elif arch == 1:
            exp_name = results_dir+"plasticc_balanced_resnet_eg"
            network = ResNet1D(nn_params)

        print(input_shape)
        print(torch.cuda.mem_get_info(device=None))
        exp_params["network_model"] =network
        
        experiment = SeededExperiment(
            exp_name,
            exp_params = exp_params,
            seeds = [1772670],
            test_data = test_dataset
            # train_data=train_dataset
            )

        experiment.run_test_phase(save_name='_batch{}_0.25'.format(i))
        del experiment.test_data
        del experiment
        torch.cuda.empty_cache()
        print(torch.cuda.mem_get_info(device=None))

    del test_dataset
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info(device=None))