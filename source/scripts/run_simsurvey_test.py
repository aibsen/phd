from math import exp
from sre_parse import expand_template
import numpy as np
import torch
import time
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
from recurrent_models import GRU1D
from convolutional_models import FCNN1D, ResNet1D
from seeded_experiment import SeededExperiment
from plot_utils import *
from transforms import MultiCropPadTensor, CropPadTensor
import analyze_simsurvey_results


results_dir = "../../results/"
training_data_dir = "../../data/ztf/training/linearly_interpolated/"
test_data_dir = "../../data/ztf/testing/"
training_fn = training_data_dir+"simsurvey_data_balanced.h5"
test_fn = test_data_dir+"simsurvey_data_balanced_1_test.h5"

lc_length = 128

num_epochs = 100
seeds = [1772670]
seed=torch.cuda.manual_seed(seed=1772670)
n_seeds = 1
num_classes = 6
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64

exp_name_template = "simsurvey_cropped_"
############ PART 1 ############### 100% LCs
# training/testing with complete light curves for all 4 models.
# train_dataset = LCs(lc_length,training_fn, n_classes=num_classes)
# train_dataset.load_data_into_memory()
# train_dataset.lens = np.full((len(train_dataset),),lc_length)

t = CropPadTensor(lc_length,cropping=0.25)
test_dataset = LCs(lc_length,test_fn, n_classes=num_classes, transforms=[t])
test_dataset.load_data_into_memory()
# print(test_dataset.lens)
test_dataset.apply_tranforms()
# test_dataset.lens = np.full((len(test_dataset),),lc_length)

input_shape = test_dataset[0][0].shape
print(input_shape)
print(len(test_dataset))
print(test_dataset.lens)

exp_params={
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "num_output_classes": num_classes,
    "patience":5,
    "validation_step":3
}

for arch in range(2,3):

    nn_params = {
        "input_shape" : input_shape,
        "num_output_classes":num_classes,
        "hidden_size" : 100
    }

    if arch>1:
        exp_name = results_dir+exp_name_template+"fixgru"
        # train_dataset.packed = True
        test_dataset.packed = True
        nn_params["attention"] = False

        if arch == 3:
            exp_name = results_dir+exp_name_template+"grusa"
            # exp_name+"sa"
            nn_params["attention"] = True
            # nn_params["r"]=1
            # nn_params["da"]=20
            
        
        network = GRU1D(nn_params)

    if arch == 0:
        exp_name = results_dir+exp_name_template+'fcn'
        network = FCNN1D(nn_params)
    elif arch == 1:
        exp_name = results_dir+exp_name_template+'resnet'
        network = ResNet1D(nn_params)

    exp_params["network_model"]=network

    try:
        experiment = SeededExperiment(
            exp_name,
            exp_params = exp_params,
            seeds = seeds,
            # train_data = train_dataset,
            test_data = test_dataset
        )
        experiment.run_test_phase(save_name='_0.25')
        analyze_simsurvey_results.overall_cm_cv(where_in_fold=exp_name)

    except Exception as e:
            print(e)
            pass