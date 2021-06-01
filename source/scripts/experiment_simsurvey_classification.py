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

"""This script is for running an experiment of classification of simsurvey lightcurves,
    using 4 different NN models: FCN, ResNet, GRU, and GRU-SA """

results_dir = "../../results/"
interpolated_dataset_filename = "../../data/training/linearly_interpolated/unbalanced_dataset_m_realzp_128_3types_small.h5"

lc_length = 128
num_epochs = 30
seed = 1772670
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64

def load_datasets(lc_length, interpolated_dataset_filename, transform=None):
    #load dataset
    interpolated_dataset = LCs(lc_length, interpolated_dataset_filename, transform=transform)
    dataset_length = len(interpolated_dataset)

    #split into train/validation/test, validation/test will be ~ .4
    val_length = int(dataset_length/4)
    test_length = int(dataset_length/4)
    train_length = dataset_length - val_length -test_length
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length, test_length])

    #dataset loaders
    print("training set length: ",str(train_length))
    print("validation set length: ",str(val_length))
    print("test set length: ",str(test_length))

    batch_size = 64

    return train_dataset, val_dataset, test_dataset, train_dataset[0][0].shape

def save_plots(results_dir,exp_name):
    validation_results = pd.read_csv(results_dir+exp_name+"/result_outputs/validation_results.csv")
    true_tagsv = validation_results.true_tags.values
    predictedv = validation_results.predicted_tags.values
    plotfile = results_dir+exp_name+"/result_outputs/validation_cm_nomalized.png"
    plot_cm(plotfile,true_tagsv,predictedv)
    plotfile = results_dir+exp_name+"/result_outputs/validation_cm.png"
    plot_cm(plotfile,true_tagsv,predictedv,False)

    test_results = pd.read_csv(results_dir+exp_name+"/result_outputs/test_results.csv")
    true_tags = test_results.true_tags.values
    predicted = test_results.predicted_tags.values
    plotfile = results_dir+exp_name+"/result_outputs/test_cm_nomalized.png"
    plot_cm(plotfile,true_tags,predicted)
    plotfile = results_dir+exp_name+"/result_outputs/test_cm.png"
    plot_cm(plotfile,true_tags,predicted,False)


train_dataset, val_dataset, test_dataset, input_shape = load_datasets(lc_length,interpolated_dataset_filename)

fcn_params = {
    "input_shape": input_shape,
    "num_output_classes" : 3,
    "regularize" : False,
    "global_pool" : 'max'
}
resnet_params = {
    "input_shape": input_shape,
    "num_output_classes" : 3,
    "global_pool":'avg',
    "n_blocks":3
}
gru_params = {
    "input_shape": input_shape,
    "num_output_classes" : 3,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"no_attention",
    "da":50,
    "r":1
    }
grusa_params = {
    "input_shape": input_shape,
    "num_output_classes" : 3,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"self_attention",
    "da":50,
    "r":1
    }

exp_names  = ["exp1_p1_fcn", "exp1_p1_resnet", "exp1_p1_gru", "exp1_p1_grusa"]
params = [fcn_params, resnet_params, gru_params, grusa_params]

exp_params={
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "num_output_classes": 3,
    "weight_decay_coefficient": wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "chunked": False
}

for m,param in enumerate(params): #for each model in the experiment 
        print(m)
        print(param)
        if m == 0: #fcn
            network = FCNN1D(param)
        elif m == 1: #resnet
            network = ResNet1D(param)
        elif m == 2: #gru
            network = GRU1D(param)
        elif m == 3: #grusa
            network = GRU1D(param)

        exp_params["network_model"] = network
        se = SeededExperiment(
            results_dir+exp_names[m],
            exp_params,
            train_data=train_dataset,
            test_data=test_dataset,
            verbose=True,
            n_seeds=1,
            seeds= [seed])

        start_time = time.time()
        se.run_experiment()
        print("--- %s seconds ---" % (time.time() - start_time))