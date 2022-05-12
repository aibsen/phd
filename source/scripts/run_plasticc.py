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
from convolutional_models import FCNN1D, ResNet1D
# from seeded_experiment import SeededExperiment
from experiment import Experiment

from torch.utils.data import RandomSampler

results_dir = "../../results/"
data_dir = "/home/ai/phd/data/plasticc/"
exp_name = "plasticc_vanilla2"


lc_length = 128
batch_size = 64
num_epochs = 10
use_gpu = True
lr = 1e-03
wdc = 1e-03
seeds = [1772670]
seed=torch.cuda.manual_seed(seed=1772670)
n_seeds = 1

exp_params={
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "chunked": False,
    "num_output_classes": 14
}


####Vanilla Plasticc

training_data_file=data_dir+'plasticc_train_data.h5'
train_dataset = LCs(lc_length, training_data_file)
input_shape = train_dataset[0][0].shape
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataset_length = len(train_dataset)
val_length = int(dataset_length/4)
test_length = int(dataset_length/4)
train_length = dataset_length - val_length -test_length
train,val,test=torch.utils.data.random_split(train_dataset, [train_length, val_length, test_length],generator=seed)
class_weights = [1/freq for freq in train_dataset.get_samples_per_class()]
# class_weights = [dataset_length/(14*freq) for freq in train_dataset.get_samples_per_class()]

fcn_params = {
    "global_pool" : 'max',
    "input_shape" : (12,128),
    
}

gru_params = {
    "hidden_size":100,
    "input_shape" : (12,128),
    }

grusa_params = {
    "hidden_size":100,
    "attention":"additive",
    "da":50,
    "r":1,

}
resnet_params = {
    "global_pool":'max',
}


params = [fcn_params, resnet_params, gru_params, grusa_params]

for m,param in enumerate(params): #for each model in the experiment 
    param["input_shape"] = input_shape
    param["num_output_classes"] = 14

    network = FCNN1D(param)
    exp_n = exp_name#+"_resnet"
    experiment = Experiment(
        network_model=network,
        experiment_name=results_dir+exp_n,
        num_epochs=10,
        train_data=train,
        val_data=val,
        test_data=test
        # class_weights = class_weights
    )
    experiment.run_experiment()
    break



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







