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
file_codes = [0,1,2]
num_output_classes = 9
patience = 5


####Vanilla Plasticc
transform1 = CropPadTensor(lc_length, cropping=0.5)
# transform1 = MultiCropPadTensor(lc_length,fractions=[0.25,0.25],croppings=[0.5,0.25])
# train_dataset = LCs(lc_length, training_data_file, transforms=[transform0,transform1])

for file_code in file_codes:

    # training_data_file=data_dir+'training/train_data_interpolated{}.h5'.format(file_code)
    # train_dataset = LCs(lc_length, training_data_file, transforms=[transform1])
    # train_dataset.load_data_into_memory()
    # input_shape = train_dataset[0][0].shape
    # train_dataset.apply_tranforms()
    # # train_dataset.packed = True
    # # train_dataset.lens=torch.full((len(train_dataset),),lc_length)


    # val_data_file=data_dir+'training/val_data_interpolated{}.h5'.format(file_code)
    # val_dataset = LCs(lc_length, val_data_file, transforms=[transform1])
    # val_dataset.load_data_into_memory()
    # val_dataset.apply_tranforms()
    # # val_dataset.packed = True
    # # val_dataset.lens=torch.full((len(val_dataset),),lc_length)

    test_data_file=data_dir+'test/test_data_interpolated{}.h5'.format(file_code)
    test_dataset = LCs(lc_length, test_data_file,transforms=[transform1])
    test_dataset.load_data_into_memory()
    input_shape = test_dataset[0][0].shape
    test_dataset.apply_tranforms()
    # test_dataset.packed = True
    # test_dataset.lens=torch.full((len(test_dataset),),int(lc_length*0.5))

    for arch in range(0,4):

        nn_params = {
            "input_shape" : input_shape,
            "num_output_classes":num_output_classes,
            "hidden_size" : 100
            # "attention" : True    
        }
        
        if arch>1:
            exp_name = results_dir+"plasticc_balanced_0_gru"
            # train_dataset.packed = True
            # val_dataset.packed = True
            test_dataset.packed = True
            # test_dataset.lens=torch.full((len(test_dataset),),int(lc_length*0.5))

            if arch == 3:
                exp_name = exp_name+"sa"
                nn_params["attention"] = True

            network = GRU1D(nn_params)

        elif arch == 0:
            exp_name = results_dir+"plasticc_balanced_0_fcn"
            network = FCNN1D(nn_params)
        elif arch == 1:
            exp_name = results_dir+"plasticc_balanced_0_resnet"
            network = ResNet1D(nn_params)

        print(input_shape)
        print(torch.cuda.mem_get_info(device=None))
        
        experiment = Experiment( 
            network,
            exp_name,
            num_epochs=num_epochs,
            num_output_classes = num_output_classes,
            learning_rate = lr,
            # train_data = train_dataset,
            # val_data = val_dataset,
            test_data = test_dataset,
            weight_decay_coefficient = wdc,
            patience = patience
            )

        # experiment.run_experiment(save_name='_{}'.format(file_code))
        experiment.run_test_phase(model_name='final_model_{}.pth.tar'.format(file_code),save_name='_0.5_3{}'.format(file_code))

    # del train_dataset
    # del val_dataset
    del test_dataset
    # del experiment.train_data
    # del experiment.val_data
    del experiment.test_data
    del experiment
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info(device=None))