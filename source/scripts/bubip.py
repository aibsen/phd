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
from seeded_experiment import SeededExperiment


data_file = "../../data/testing/real_data_30_careful.h5"
dataset = LCs(128, data_file)
seed = 1772670
torch.manual_seed(seed=seed)
train_dataset = torch.utils.data.ConcatDataset([dataset,dataset,dataset,dataset,dataset])

gru_params = {
    "input_shape": dataset[0][0].shape,
    "num_output_classes" : 4,
    "hidden_size":10,
    "batch_size":64,
    "attention":"self_attention",
    "da":20,
    "r":3
    }

gru = GRU1D(gru_params)

exp_params={
    "network_model":gru,
    "num_epochs": 5,
    "learning_rate": 1e-03,
    "weight_decay_coefficient": 1e-03,
    "use_gpu": True,
    "batch_size":64,
    "balance_training_set" :True
}




se = SeededExperiment("../../results/s_exp_dummy", exp_params,
    train_data=train_dataset,test_data=dataset,verbose=False,n_seeds=5,seeds=[1,2,3,4,5])
se.run_experiment()