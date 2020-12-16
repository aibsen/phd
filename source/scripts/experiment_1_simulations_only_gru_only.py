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

#this experiment is for data simulation only. It has 4 parts:
#part 1: training/testing with complete light curves for all 4 models.
#part 2 : training/testing with half light curves for all 4 models, using the 1st half
#part 3 : training/testing with half light curves for all 4 models, random chunks
#part 4 : training using complete light curves and randomly cropped light curves (padded), 
#testing using padded light curves chopped randomly.
#for all of these, lc length = 128 and training is unbalanced


results_dir = "../../results/"
interpolated_dataset_filename = "../../data/training/linearly_interpolated/unbalanced_dataset_m_realzp_128.h5"

lc_length = 128
lc_transform_length = 64

random_crop=RandomCrop(lc_transform_length,lc_length)

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

############ PART 1 ############### 100% LCs
# training/testing with complete light curves for all 4 models.

#1.D RNN-attention
train_dataset, val_dataset, test_dataset, input_shape = load_datasets(int(lc_length), interpolated_dataset_filename)

grusa_params = {
    "input_shape": input_shape,
    "num_output_classes" : 4,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"self_attention",
    "da":75,
    "r":5
    }

exp_name = "exp1_p1 _grusa_r2_d75"
#define network parameters

grusa = GRU1D(grusa_params)

experiment = Experiment(
    network_model = grusa,
    experiment_name = results_dir+exp_name,
    num_epochs = num_epochs,
    learning_rate = lr,
    weight_decay_coefficient = wdc,
    use_gpu = use_gpu,
    train_data = train_dataset,
    val_data = val_dataset,
    test_data = test_dataset
)
start_time = time.time()
experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
save_plots(results_dir,exp_name)
print("--- %s seconds ---" % (time.time() - start_time))



############ PART 2 ############### 50%
# training/testing with half light curves for all 4 models, using the 1st half

#load dataset
train_dataset, val_dataset, test_dataset, input_shape = load_datasets(int(lc_length/2), interpolated_dataset_filename)


#2.D RNN-attention
exp_name = "exp1_p2_grusa_r2_d75"
#define network parameters
grusa_params["input_shape"]=input_shape
grusa = GRU1D(grusa_params)

experiment = Experiment(
    network_model = grusa,
    experiment_name = results_dir+exp_name,
    num_epochs = num_epochs,
    learning_rate = lr,
    weight_decay_coefficient = wdc,
    use_gpu = use_gpu,
    train_data = train_dataset,
    val_data = val_dataset,
    test_data = test_dataset
)
start_time = time.time()
experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
save_plots(results_dir,exp_name)
print("--- %s seconds ---" % (time.time() - start_time))


############ PART 3 ############### 50% (former part 9)
# training using complete light curves and cropped light curves (1st bits) (padded), 
# testing using padded light curves chopped. 50% of lcs

lc_transform_length = 64
zeropad = ZeroPad(lc_length,lc_transform_length)
right_crop=RightCrop(lc_transform_length,lc_length)
composed = transforms.Compose([right_crop,zeropad])

#load dataset
interpolated_dataset = LCs(lc_length, interpolated_dataset_filename)
dataset_length = len(interpolated_dataset)
#split into train/validation/test, validation/test will be ~ .4
val_length = int(dataset_length/4)
test_length = int(dataset_length/4)
train_length = dataset_length - val_length -test_length
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length, test_length])
input_shape = train_dataset[0][0].shape
train_length1 = int(train_length/2)
train_length2 = train_length - train_length1
train_dataset1, train_dataset2 = torch.utils.data.random_split(train_dataset, [train_length1,train_length2])
#apply transforms to half of train, val and test
train_dataset1.transform=composed
val_dataset.transform=composed
test_dataset.transform=composed
train_dataset = torch.utils.data.ConcatDataset([train_dataset1,train_dataset2])
input_shape = train_dataset[0][0].shape

#3.D RNN-attention
exp_name = "exp1_p3_grusa_r2_d75"
grusa_params["input_shape"]=input_shape
grusa = GRU1D(grusa_params)

experiment = Experiment(
    network_model = grusa,
    experiment_name = results_dir+exp_name,
    num_epochs = num_epochs,
    learning_rate = lr,
    weight_decay_coefficient = wdc,
    use_gpu = use_gpu,
    train_data = train_dataset,
    val_data = val_dataset,
    test_data = test_dataset
)
start_time = time.time()
experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
save_plots(results_dir,exp_name)
print("--- %s seconds ---" % (time.time() - start_time))

############ PART 4 ############### 25% (former part 5)
# training/testing with 1/4 light curves for all 4 models, using the 1st half

#load dataset
train_dataset, val_dataset, test_dataset, input_shape =load_datasets(int(lc_length/4), interpolated_dataset_filename)

#4.D RNN-attention
exp_name = "exp1_p4_grusa_r2_d75"
#define network parameters
grusa_params["input_shape"]=input_shape
grusa = GRU1D(grusa_params)

experiment = Experiment(
    network_model = grusa,
    experiment_name = results_dir+exp_name,
    num_epochs = num_epochs,
    learning_rate = lr,
    weight_decay_coefficient = wdc,
    use_gpu = use_gpu,
    train_data = train_dataset,
    val_data = val_dataset,
    test_data = test_dataset
)
start_time = time.time()
experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
save_plots(results_dir,exp_name)
print("--- %s seconds ---" % (time.time() - start_time))


############ PART 5 ############### former part 10
#training using complete light curves and cropped light curves (1st bits) (padded), 
#testing using padded light curves chopped. 25% of lcs

lc_transform_length = 32
zeropad = ZeroPad(lc_length,lc_transform_length)
right_crop=RightCrop(32,lc_length)
composed = transforms.Compose([right_crop,zeropad])

#load dataset
interpolated_dataset = LCs(lc_length, interpolated_dataset_filename)
dataset_length = len(interpolated_dataset)
#split into train/validation/test, validation/test will be ~ .4
val_length = int(dataset_length/4)
test_length = int(dataset_length/4)
train_length = dataset_length - val_length -test_length
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length, test_length])
input_shape = train_dataset[0][0].shape
train_length1 = int(train_length/2)
train_length2 = train_length - train_length1
train_dataset1, train_dataset2 = torch.utils.data.random_split(train_dataset, [train_length1,train_length2])
#apply transforms to half of train, val and test
train_dataset1.transform=composed
val_dataset.transform=composed
test_dataset.transform=composed
train_dataset = torch.utils.data.ConcatDataset([train_dataset1,train_dataset2])
input_shape = train_dataset[0][0].shape

#5.D RNN-attention
exp_name = "exp1_p5_grusa_r2_d75"
grusa_params["input_shape"]=input_shape
grusa = GRU1D(grusa_params)

experiment = Experiment(
    network_model = grusa,
    experiment_name = results_dir+exp_name,
    num_epochs = num_epochs,
    learning_rate = lr,
    weight_decay_coefficient = wdc,
    use_gpu = use_gpu,
    train_data = train_dataset,
    val_data = val_dataset,
    test_data = test_dataset
)
start_time = time.time()
experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
save_plots(results_dir,exp_name)
print("--- %s seconds ---" % (time.time() - start_time))

############ PART 6 ############### former part 11
#training using complete light curves and cropped light curves (1st bits) (50% + 25%) (padded), 
#testing using padded light curves chopped. 25% of lcs

zeropad_32 = ZeroPad(lc_length,32)
zeropad_64 = ZeroPad(lc_length,64)
right_crop_32=RightCrop(32,lc_length)
right_crop_64=RightCrop(64,lc_length)
composed_32 = transforms.Compose([right_crop_32,zeropad_32])
composed_64 = transforms.Compose([right_crop_64,zeropad_64])

#load dataset
interpolated_dataset = LCs(lc_length, interpolated_dataset_filename)
dataset_length = len(interpolated_dataset)
#split into train/validation/test, validation/test will be ~ .4
val_length = int(dataset_length/4)
test_length = int(dataset_length/4)
train_length = dataset_length - val_length -test_length
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length, test_length])
input_shape = train_dataset[0][0].shape
train_length1 = int(train_length/3)
train_length2 = int(train_length/3)
train_length3 = train_length - train_length1 - train_length2
train_dataset1, train_dataset2, train_dataset3 = torch.utils.data.random_split(train_dataset, [train_length1,train_length2, train_length3])
#apply transforms to 2/3 of train, val and test
train_dataset1.transform=composed_32
train_dataset2.transform=composed_64
val_dataset.transform=composed_32
test_dataset.transform=composed_32
train_dataset = torch.utils.data.ConcatDataset([train_dataset1,train_dataset2, train_dataset3])
input_shape = train_dataset[0][0].shape


#6.D RNN-attention
exp_name = "simonly_p6_grusa"
grusa_params["input_shape"]=input_shape
grusa = GRU1D(grusa_params)

experiment = Experiment(
    network_model = grusa,
    experiment_name = results_dir+exp_name,
    num_epochs = num_epochs,
    learning_rate = lr,
    weight_decay_coefficient = wdc,
    use_gpu = use_gpu,
    train_data = train_dataset,
    val_data = val_dataset,
    test_data = test_dataset
)
start_time = time.time()
experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
save_plots(results_dir,exp_name)
print("--- %s seconds ---" % (time.time() - start_time))