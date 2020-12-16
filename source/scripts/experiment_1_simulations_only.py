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
interpolated_dataset_filename = "../../data/training/linearly_interpolated/unbalanced_dataset_m_realzp_128_3types_small.h5"

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

#1.A FCN
# exp_name = "simonly_p1_fcn"
# train_dataset, val_dataset, test_dataset, input_shape = load_datasets(lc_length,interpolated_dataset_filename)

# fcn_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 3,
#     "regularize" : False,
#     "global_pool" : 'max'
# }

# fcn = FCNN1D(fcn_params)

# experiment = Experiment(
#     network_model = fcn,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))

# #1.B ResNet
# exp_name = "simonly_p1_resnet"

# resnet_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 3,
#     "global_pool":'avg',
#     "n_blocks":3
# }

# resnet = ResNet1D(resnet_params)

# experiment = Experiment(
#     network_model = resnet,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))

# #1.C RNN
# exp_name = "simonly_p1_gru"
# #define network parameters
# gru_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 3,
#     "hidden_size":100,
#     "batch_size":batch_size,
#     "attention":"no_attention",
#     "da":50,
#     "r":1
#     }

# gru = GRU1D(gru_params)

# experiment = Experiment(
#     network_model = gru,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))


# #1.D RNN-attention
# exp_name = "simonly_p1_grusa"
# #define network parameters
# grusa_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 3,
#     "hidden_size":100,
#     "batch_size":batch_size,
#     "attention":"self_attention",
#     "da":50,
#     "r":1
#     }

# grusa = GRU1D(grusa_params)

# experiment = Experiment(
#     network_model = grusa,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))



############ PART 2 ############### 50%
# training/testing with half light curves for all 4 models, using the 1st half

#load dataset
# train_dataset, val_dataset, test_dataset, input_shape = load_datasets(int(lc_length/2), interpolated_dataset_filename)

# fcn_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 3,
#     "regularize" : False,
#     "global_pool" : 'max'
# }

# resnet_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 3,
#     "global_pool":'avg',
#     "n_blocks":3
# }

# gru_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 3,
#     "hidden_size":100,
#     "batch_size":batch_size,
#     "attention":"no_attention",
#     "da":50,
#     "r":1
#     }

# grusa_params = {
#     "input_shape": input_shape,
#     "num_output_classes" : 3,
#     "hidden_size":100,
#     "batch_size":batch_size,
#     "attention":"self_attention",
#     "da":50,
#     "r":1
#     }


# #2.A FCN
# exp_name = "simonly_p2_fcn"
# fcn_params["input_shape"] = input_shape
# fcn = FCNN1D(fcn_params)

# experiment = Experiment(
#     network_model = fcn,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))

# #2.B ResNet
# exp_name = "simonly_p2_resnet"
# resnet_params["input_shape"] = input_shape
# resnet = ResNet1D(resnet_params)

# experiment = Experiment(
#     network_model = resnet,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))

# #2.C RNN
# exp_name = "simonly_p2_gru"
# gru_params["input_shape"] = input_shape
# gru = GRU1D(gru_params)

# experiment = Experiment(
#     network_model = gru,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))


# #2.D RNN-attention
# exp_name = "simonly_p2_grusa"
# #define network parameters
# grusa_params["input_shape"]=input_shape
# grusa = GRU1D(grusa_params)

# experiment = Experiment(
#     network_model = grusa,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))


############ PART 3 ############### 50%
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

batch_size = 64
# #3.A FCN
# exp_name = "simonly_p3_fcn"
# fcn_params["input_shape"] = input_shape
# fcn = FCNN1D(fcn_params)

# experiment = Experiment(
#     network_model = fcn,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))

# #3.B ResNet
# exp_name = "simonly_p3_resnet"
# resnet_params["input_shape"] = input_shape
# resnet = ResNet1D(resnet_params)

# experiment = Experiment(
#     network_model = resnet,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))

# #3.C RNN
# exp_name = "simonly_p3_gru"
# gru_params["input_shape"] = input_shape
# gru = GRU1D(gru_params)

# experiment = Experiment(
#     network_model = gru,
#     experiment_name = results_dir+exp_name,
#     num_epochs = num_epochs,
#     learning_rate = lr,
#     weight_decay_coefficient = wdc,
#     use_gpu = use_gpu,
#     train_data = train_dataset,
#     val_data = val_dataset,
#     test_data = test_dataset
# )
# start_time = time.time()
# experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
# save_plots(results_dir,exp_name)
# print("--- %s seconds ---" % (time.time() - start_time))


#3.D RNN-attention
exp_name = "simonly_p3_grusa"
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

############ PART 4 ############### 25%
# training/testing with 1/4 light curves for all 4 models, using the 1st half

#load dataset
train_dataset, val_dataset, test_dataset, input_shape =load_datasets(int(lc_length/4), interpolated_dataset_filename)

#4.A FCN
exp_name = "simonly_p4_fcn"
fcn_params["input_shape"] = input_shape
fcn = FCNN1D(fcn_params)

experiment = Experiment(
    network_model = fcn,
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

#4.B ResNet
exp_name = "simonly_p4_resnet"
resnet_params["input_shape"] = input_shape
resnet = ResNet1D(resnet_params)

experiment = Experiment(
    network_model = resnet,
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

#4.C RNN
exp_name = "simonly_p4_gru"
gru_params["input_shape"] = input_shape
gru = GRU1D(gru_params)

experiment = Experiment(
    network_model = gru,
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


#4.D RNN-attention
exp_name = "simonly_p4_grusa"
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


############ PART 5 ###############
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

#5.A FCN
exp_name = "simonly_p5_fcn"
fcn_params["input_shape"] = input_shape
fcn = FCNN1D(fcn_params)

experiment = Experiment(
    network_model = fcn,
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

#5.B ResNet
exp_name = "simonly_p5_resnet"
resnet_params["input_shape"] = input_shape
resnet = ResNet1D(resnet_params)

experiment = Experiment(
    network_model = resnet,
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

#5.C RNN
exp_name = "simonly_p5_gru"
gru_params["input_shape"] = input_shape
gru = GRU1D(gru_params)

experiment = Experiment(
    network_model = gru,
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


#5.D RNN-attention
exp_name = "simonly_p5_grusa"
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

############ PART 6 ###############
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

#6.A FCN
exp_name = "simonly_p6_fcn"
fcn_params["input_shape"] = input_shape
fcn = FCNN1D(fcn_params)

experiment = Experiment(
    network_model = fcn,
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

#6.B ResNet
exp_name = "simonly_p6_resnet"
resnet_params["input_shape"] = input_shape
resnet = ResNet1D(resnet_params)

experiment = Experiment(
    network_model = resnet,
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

#6.C RNN
exp_name = "simonly_p6_gru"
gru_params["input_shape"] = input_shape
gru = GRU1D(gru_params)

experiment = Experiment(
    network_model = gru,
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