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

lc_lengths = [24,32]
arch_code = 1
# noise_code = [4,5,6,7]
noise_code = [4]
binning_codes = ['73','70','03']

results_dir = "../../results/"
data_dir = "/home/ai/phd/data/plasticc/dmdt/training/"

# exp_name = results_dir+"dummy"

batch_size = 64
num_epochs = 100
use_gpu = True
lr = 1e-03
wdc = 1e-02
seeds = [1772670]#, 12345, 160291]

for lc_length in lc_lengths:
    for b_code in binning_codes:
        for noise in noise_code:
            # exp_name = results_dir+"{}_{}x{}_b{}".format(arch_code,lc_length,lc_length,b_code)
            exp_name = results_dir+"{}_{}x{}_b{}augmented_noise{}".format(arch_code,lc_length,lc_length,b_code,noise)
            training_data_file=data_dir+'dmdts_training_{}x{}_b{}augmented_noise{}.h5'.format(lc_length,lc_length,b_code,noise)
            # training_data_file=data_dir+'dmdts_training_{}x{}_b{}.h5'.format(lc_length,lc_length,b_code)
            print(training_data_file)
            train_dataset = LCs(lc_length,training_data_file,n_channels=6)
            train_dataset.load_data_into_memory()

        # dataset_length = len(train_dataset)
        # val_length = int(dataset_length/4)
        # train_length = dataset_length - val_length
        # train, val=torch.utils.data.random_split(train_dataset, [train_length, val_length],generator=seed)

            print(len(train_dataset))
            input_shape = train_dataset[0][0].shape
            print(input_shape)

            ####DMDT Plasticc
            cnn_params={
            'input_shape': input_shape,
            # 'n_filters': 64,
            'kernel_size':3,
            "num_output_classes": 14
            #  "n_layers":2
            }
            if arch_code == 0:
                cnn_params['n_filters'] = 32
                network = DMDTShallowCNN(cnn_params)

            elif arch_code == 1:
                cnn_params['n_filters'] = 64
                network = DMDTCNN(cnn_params)

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

            experiment = SeededExperiment(
                exp_name = exp_name,
                exp_params = exp_params,
                seeds = seeds,
                train_data = train_dataset
            )

            experiment.run_experiment()

    # experiment = Experiment(
    #     experiment_name = exp_name,
    #     network_model = network,
    #     train_data = train,
    #     val_data = val,
    #     test_data = test,
    #     num_epochs = num_epochs
    # )

    # experiment.run_experiment()
