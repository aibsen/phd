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
from transforms import GroupClass, GroupClassTensor
from convolutional_models import FCNN1D, ResNet1D
from recurrent_models import GRU1D

results_dir = "../../results/"
data_dir = "../../data/plasticc/interpolated/"
exp_name = results_dir+"plasticc_test_grusa_eg"


lc_length = 128
batch_size = 64
num_epochs = 100
use_gpu = True
lr = 1e-04
wdc = 1e-02
seeds = [1772670]
# seed=torch.cuda.manual_seed(seed=1772670)
n_seeds = 1
num_classes = 9 

####Vanilla Plasticc
training_data_file=data_dir+'test/plasticc_test_data_batch2_extragalactic.h5'
# training_data_file=data_dir+'training/plasticc_train_data_augmented_extragalactic.h5'
train_dataset = LCs(lc_length, training_data_file, n_classes=num_classes)
train_dataset.load_data_into_memory()
# train_dataset.apply_tranforms()
input_shape = train_dataset[0][0].shape

#for rnns
train_dataset.packed=True
train_dataset.lens=torch.full((len(train_dataset),),lc_length)

print(len(train_dataset))


nn_params = {
    "input_shape" : input_shape,
    "num_output_classes":num_classes,
    "hidden_size" : 100,
    "attention" : True    
}

network = GRU1D(nn_params)

exp_params={
    "network_model": network,
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "num_output_classes": num_classes,
    "patience":5,
    "validation_step":3
}

experiment = SeededExperiment( 
    exp_name,
    exp_params = exp_params,
    seeds = [1772670],
    train_data=train_dataset
    )

experiment.run_experiment()


for i in range(3,12):
    test_data_file = data_dir+'test/plasticc_test_data_batch{}_extragalactic.h5'.format(i)
    test_dataset = LCs(lc_length, test_data_file)
    print("loading dataset")
    test_dataset.load_data_into_memory()
#     test_dataset.apply_tranforms()
#for rnns
    test_dataset.packed = True
    test_dataset.lens=torch.full((len(test_dataset),),lc_length)

    print(torch.cuda.mem_get_info(device=None))
    experiment.test_data = test_dataset
    experiment.run_test_phase(test_data_name='test_batch{}'.format(i))
    del test_dataset
    del experiment.test_data
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info(device=None))

  