import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs, CachedLCs 
from transforms import RandomCrop,ZeroPad,RightCrop
from recurrent_models import GRU1D
from convolutional_models import FCNN1D, ResNet1D
from experiment import Experiment
from plot_utils import *
from torchvision import transforms
from seeded_experiment import SeededExperiment


results_dir = "../../results/"
interpolated_dataset_filename = "../../data/training/linearly_interpolated/unbalanced_dataset_m_realzp_128_small.h5"

lc_length = 128
num_epochs = 30
seed = 77777
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64

grusa_params = {
    "num_output_classes" : 4,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"self_attention",
    "da":50,
    "r":1
    }


#load dataset
interpolated_dataset = LCs(lc_length, interpolated_dataset_filename)
dataset_length = len(interpolated_dataset)
#split into train/validation/test, validation/test will be ~ .4
# val_length = int(dataset_length/4)
test_length = int(dataset_length/3)
train_length = dataset_length -test_length
train_dataset, test_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, test_length])
input_shape = train_dataset[0][0].shape


def run_exp(model,name):

    exp_params={
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "num_output_classes": 4,
    "weight_decay_coefficient": wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "chunked": False,
    "network_model":model
}
    experiment = SeededExperiment(
        exp_name = results_dir+name,
        exp_params=exp_params,
        train_data = train_dataset,
        test_data = test_dataset,
        n_seeds=1,
        seeds= [seed],
        k=3
    )
    start_time = time.time()
    experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
    print("--- %s seconds ---" % (time.time() - start_time))



############ PART 1 ############### tuning of r
for i in np.arange(1,6):
    for j in np.arange(25,125,25):
        exp_name = "4c_cv_sa_tunning_grusa_r"+str(i)+"_d_"+str(j)
        print(exp_name)
        grusa_params["input_shape"]=input_shape
        grusa_params["r"]=i
        grusa_params["da"]=j
        grusa = GRU1D(grusa_params)
        run_exp(grusa,exp_name)


# # ############ PART 2 ############### tuning of da
# grusa_params["r"]=4 #best of the above

#     exp_name = "4c_sa_tunning_grusa_r4_da"+str(j)
#     grusa_params["input_shape"]=input_shape
#     grusa_params["da"]=j
#     grusa = GRU1D(grusa_params)
#     run_exp(grusa,exp_name)
