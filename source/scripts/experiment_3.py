import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import h5py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs, CachedLCs, RandomCrop,ZeroPad,RightCrop
from recurrent_models import GRU1D
from convolutional_models import FCNN1D, ResNet1D
from experiment import Experiment
from plot_utils import *
from torchvision import transforms

#this experiment is for testing real data only, woth at least 40 days observations It has several parts:
#part 1: classifiers were trained with complete simulated light curves for all 4 models. lc_len = 128
#part 2 : classifiers were trained with half light curves for all 4 models, using the 1st half. lc_len = 64
#part 3 : classifiers were trained with half light curves for all 4 models, random chunks. lc_len = 64
#part 4 : classifiers were trained with complete light curves and randomly cropped half light curves (padded). lc_len = 128
#part 5 : classifiers were trained with 1/4 lcs 1st bit. lc_len = 32
#part 6 : classifiers were trained with 1/4 lcs random bit. lc_len = 32
#part 7 : classifiers were trained with complete lcs and 1/4 lcs random bit. lc_len = 128
#part 8 : classifiers were trained with complete lcs and random 10% bit. lc_len = 128
#part 9 : classifiers were trained with complete lcs and 1st half. lc_len = 128
#part 10 : classifiers were trained with complete lcs and 1st 1/4. lc_len = 128
#part 11 : classifiers were trained with complete lcs and 1st half + 1/4. lc_len = 128

results_dir = "../../results/"
interpolated_dataset_filename = "../../data/training/linearly_interpolated/unbalanced_dataset_m_realzp_128.h5"

lc_length = 128
num_epochs = 100
seed = 1772670
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64


fcn_params = {
    "num_output_classes" : 4,
    "regularize" : False,
    "global_pool" : 'max'
}
resnet_params = {
    "num_output_classes" : 4,
    "global_pool":'avg',
    "n_blocks":3
}
gru_params = {
    "num_output_classes" : 4,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"no_attention",
    "da":50,
    "r":1
    }
grusa_params = {
    "num_output_classes" : 4,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"self_attention",
    "da":50,
    "r":1
    }
params = [fcn_params, resnet_params, gru_params, grusa_params]

test_dataset_filename = "../../data/testing/real_data_30_careful.h5"
lc_lens = [128,64,64,128,32,32,128,128,128,128,128] #length input shape of trained classifiers
results_dir = "../../results/"

def load_dataset(lc_length, test_dataset_filename, transform=None):
    test_dataset = LCs(lc_length, test_dataset_filename, transform=transform)
    test_length = len(test_dataset)
    print("test set length: ",str(test_length))
    print(test_dataset[0][0].shape)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)
    return test_loader, test_dataset[0][0].shape

def find_best_epoch(f):
    results_summary = pd.read_csv(f)
    val_f1 = results_summary.val_f1.values
    best_epoch = val_f1.argmax()
    return best_epoch

for i in np.arange(7,12): #for each experiment

    exp_names = list(map(lambda x: x.format(i),["exp1_p{}_fcn", "exp1_p{}_resnet", "exp1_p{}_gru", "exp1_p{}_grusa"]))
    for m,param in enumerate(params): #for each model in the experiment 
        best_epoch = find_best_epoch(results_dir+exp_names[m]+"/result_outputs/summary.csv")

        test_loader, input_shape = load_dataset(lc_lens[i-1],test_dataset_filename)
        param["input_shape"] = input_shape

        if m == 0: #fcn
            network = FCNN1D(param)
        elif m == 1: #resnet
            network = ResNet1D(param)
        elif m == 2: #gru
            network = GRU1D(param)
        elif m == 3: #grusa
            network = GRU1D(param)


        experiment = Experiment(
            network_model = network,
            experiment_name = results_dir+exp_names[m],
            num_epochs = num_epochs,
            learning_rate = lr,
            weight_decay_coefficient = wdc,
            use_gpu = use_gpu,
            test_data = test_loader,
            best_idx = best_epoch
        )
        start_time = time.time()
        test_results_filename= "test_results_real_30_careful.csv"
        test_results_summary_filename = "test_results_real_summary_30_careful.csv"
        experiment.run_test_phase(test_loader,test_results_filename,test_results_summary_filename)
        print("--- %s seconds ---" % (time.time() - start_time))

    