import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import CachedLCs, LCs
from data_samplers import CachedRandomSampler
from experiment import Experiment
# from plot_utils import *
from torchvision import transforms
from transforms import RandomCrop,ZeroPad,RightCrop,RandomCropsZeroPad
from recurrent_models import GRU1D
from convolutional_models import CNN
from dmdt_cnns import DMDTShallowCNN, DMDTCNN, VanillaCNN
from seeded_experiment import SeededExperiment
from experiment import Experiment 
import matplotlib.pyplot as plt

from torch.utils.data import RandomSampler

lc_length = 24
arch_code = 2
# noise_code = [4,5,6,7]
noise = 4
b_code = '00'

results_dir = "../../results/"
data_dir_training = "/home/ai/phd/data/plasticc/dmdt/training/"
data_dir_test = "/home/ai/phd/data/plasticc/dmdt/test/"

# exp_name = results_dir+"dummy"

batch_size = 64
num_epochs = 60
use_gpu = True
lr = 1e-04
lr_str='04'
wdc = 1e-02
wdc_str='02'
seeds = [1772670]#, 12345, 160291]

n_conv_layers = 1
# kernel_sizes = [1,2,3,4,5]
kernel_size = 5
# kernel_size = {'0':4,'1':2} 
kernel_size_str= str(kernel_size)
# kernel_size_str='4-3'
# n_filterss = [24,32,40,48,64]
n_filters = 32
# pool_sizes = [1,2,3,4]
pool_size = 2
# drop_out_convs = [0.1,0.2,0.25,0.3]
drop_out_conv = 0.25
# drop_out_linears = [0.1,0.2,0.25,0.3]
drop_out_linear = 0.25
# out_linears = [24,32,64,128,256]
out_linear = 128

exp_name = results_dir+"{}_{}x{}final".format(arch_code,lc_length,lc_length)
training_data_file=data_dir_training+'dmdts_training_{}x{}_b{}augmented_noise{}.h5'.format(lc_length,lc_length,b_code,noise)
print(training_data_file)
# train_dataset = LCs(lc_length,training_data_file,n_channels=6)
# train_dataset.load_data_into_memory()
input_shape = torch.Size([6, 24, 24])

####DMDT Plasticc
cnn_params={
'input_shape': input_shape,
'num_output_classes': 14,
'n_conv_layers': n_conv_layers,
'kernel_size': kernel_size,
'n_filters': n_filters,
'pool_size': pool_size,
'drop_out_conv':drop_out_conv,
'drop_out_linear':drop_out_linear,
'out_linear':out_linear            
}

network = VanillaCNN(cnn_params)

exp_params={
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "chunked": False,
    "num_output_classes": 14,
    "network_model":network,
    "patience":5,
    "validation_step":3
}

    
experiment = Experiment(
    network_model = exp_params["network_model"],
    experiment_name = exp_name,
    num_epochs = exp_params["num_epochs"],
    learning_rate = exp_params["learning_rate"],
    weight_decay_coefficient = exp_params["weight_decay_coefficient"],
    batch_size = exp_params["batch_size"],
    num_output_classes=exp_params["num_output_classes"],
)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# experiment.run_final_train_phase([train_loader])
experiment.load_model(exp_name+"/saved_models","final_model.pth.tar")

for i in range(7,8):
    test_data_file=data_dir_test+'dmdts_test_{}x{}_b{}{}_no99.h5'.format(lc_length,lc_length,b_code,i)
    print(test_data_file)
    test_dataset = LCs(lc_length,test_data_file,n_channels=6)
    test_dataset.load_data_into_memory()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    experiment.test_data = test_loader
    experiment.run_test_phase(test_loader, load_model=False, data_name="test_{}".format(i))



