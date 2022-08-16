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
from seeded_experiment import SeededExperiment
from transforms import CropPadTensor, GroupClass, GroupClassTensor, MultiCropPadTensor
from convolutional_models import FCNN1D, ResNet1D
from recurrent_models import GRU1D

results_dir = "../../results/"
data_dir = "../../data/plasticc/interpolated/"
exp_name = results_dir+"plasticc_cropped_gru_eg"


lc_length = 128
batch_size = 64
num_epochs = 100
use_gpu = True
lr = 1e-04
wdc = 1e-02
seeds = [1772670]
# seed=torch.cuda.manual_seed(seed=1772670)
n_seeds = 1


####Vanilla Plasticc
# training_data_file=data_dir+'training/plasticc_train_data.h5'
training_data_file=data_dir+'test/plasticc_test_data_batch2_extragalactic.h5'
transform0 = GroupClassTensor(14,14)
transform1 = MultiCropPadTensor(lc_length,fractions=[0.25,0.25],croppings=[0.5,0.25])
train_dataset = LCs(lc_length, training_data_file, transforms=[transform0,transform1])
train_dataset.load_data_into_memory()
input_shape = train_dataset[0][0].shape
train_dataset.apply_tranforms()
train_dataset.packed = True
# print(len(train_dataset))
# torch.Size([12, 128])
print(input_shape)

nn_params = {
    "input_shape" : input_shape,
    "num_output_classes":15,
    "hidden_size" : 100
    # "attention" : True    
}


# data_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
network = GRU1D(nn_params).to(torch.device('cuda'))


# for (x,y,ids) in data_loader:
    # print(x)
    # network.forward(x)
    # break
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
    seeds = [1772670],
    train_data=train_dataset
    )

experiment.run_experiment()


transform2 = CropPadTensor(lc_length,cropping=0.25)

for i in range(3,12):
    test_data_file = data_dir+'test/plasticc_test_data_batch{}_extragalactic.h5'.format(i)
    test_dataset = LCs(lc_length, test_data_file, transforms=[transform0, transform2])
    print("loading dataset")
    test_dataset.load_data_into_memory()
    test_dataset.apply_tranforms()
    test_dataset.packed = True
    print(torch.cuda.mem_get_info(device=None))
    experiment.test_data = test_dataset
    experiment.run_test_phase(test_data_name='test_batch{}_0.25'.format(i))
    del test_dataset
    del experiment.test_data
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info(device=None))

  