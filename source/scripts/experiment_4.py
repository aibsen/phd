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


results_dir = "../../results/"
interpolated_dataset_filename = "../../data/training/linearly_interpolated/unbalanced_dataset_m_realzp_328.h5"
real_dataset_filename = "../../data/testing/real_data_30do_careful.h5"

lc_length = 328
num_epochs = 100
seed = 1772670
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64

############ PART 1 ###############
#training using complete simulated light curves and cropped light curves (1st bits) (50% + 25%) (padded) + real lcs that were classified correctly 
#testing using padded light curves chopped. 25% of lcs

def get_classified_and_missclassified_lcs():
    part = 11
    exp_names = list(map(lambda x: x.format(part),["exp1_p{}_fcn", "exp1_p{}_resnet", "exp1_p{}_gru", "exp1_p{}_grusa"]))
    filenames = list(map(lambda x: results_dir+x+"/result_outputs/test_results_real_30_careful.csv",exp_names))
    id_set = set()
    for f in filenames:
        test_results = pd.read_csv(f)
        rights = test_results[test_results.true_tags==test_results.predicted_tags]
        rights_ids = np.sort(rights.id.values)
        id_set.update(rights_ids)

    print(len(id_set))
    real_dataset = LCs(lc_length, real_dataset_filename)
    print(real_dataset[34][0].shape)
    print(len(real_dataset))



# #load dataset
interpolated_dataset = LCs(lc_length, interpolated_dataset_filename)
test_dataset = LCs(lc_length, real_dataset_filename)
dataset_length = len(interpolated_dataset)
test_length = len(test_dataset)
#split into train/validation/test, validation/test will be ~ .4
val_length = int(dataset_length/4)
train_length = dataset_length - val_length
train_dataset, val_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length])

if train_dataset[0][0].shape == test_dataset[0][0].shape:
    train_length1 = int(train_length/4)
    train_length2 = int(train_length/4)
    train_length3 = int(train_length/4)
    train_length4 = train_length - train_length1 - train_length2 - train_length3
    trd1, trd2, trd3, trd4  = torch.utils.data.random_split(train_dataset, [train_length1,train_length2, train_length3, train_length4])

    #apply transforms so we'll be training with 100, 50, 25 and 10 percent of light curves

    for p,trd in list(zip([0.1,0.25,0.5],[trd1,trd2,trd3])):
        crop=RandomCrop(int(lc_length*p),lc_length)
        zeropad = ZeroPad(lc_length,int(lc_length*p))
        composed = transforms.Compose([crop,zeropad])
        trd.transform = composed

    train_dataset = torch.utils.data.ConcatDataset([trd1,trd2,trd3,trd4])

    #dataset loaders
    print("training set length: ",str(train_length))
    print("validation set length: ",str(val_length))
    print("test set length: ",str(test_length))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    input_shape = train_dataset[0][0].shape

    fcn_params = {
        "input_shape": input_shape,
        "num_output_classes" : 4,
        "regularize" : False,
        "global_pool" : 'max'
    }
    resnet_params = {
        "input_shape": input_shape,
        "num_output_classes" : 4,
        "global_pool":'avg',
        "n_blocks":3
    }
    gru_params = {
        "input_shape": input_shape,
        "num_output_classes" : 4,
        "hidden_size":100,
        "batch_size":batch_size,
        "attention":"no_attention",
        "da":50,
        "r":1
        }
    grusa_params = {
        "input_shape": input_shape,
        "num_output_classes" : 4,
        "hidden_size":100,
        "batch_size":batch_size,
        "attention":"self_attention",
        "da":50,
        "r":1
        }


    # #2.A FCN
    # exp_name = "exp2_p1_fcn"
    # fcn = FCNN1D(fcn_params)

    # experiment = Experiment(
    #     network_model = fcn,
    #     experiment_name = results_dir+exp_name,
    #     num_epochs = num_epochs,
    #     learning_rate = lr,
    #     weight_decay_coefficient = wdc,
    #     use_gpu = use_gpu,
    #     train_data = train_loader,
    #     val_data = val_loader,
    #     test_data = test_loader
    # )
    # start_time = time.time()
    # experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
    # save_plots(results_dir,exp_name)
    # print("--- %s seconds ---" % (time.time() - start_time))

    #2.B ResNet
    exp_name = "exp1_p2_resnet"
    resnet_params["input_shape"] = input_shape
    resnet = ResNet1D(resnet_params)

    experiment = Experiment(
        network_model = resnet,
        experiment_name = results_dir+exp_name,
        num_epochs = num_epochs,
        learning_rate = lr,
        weight_decay_coefficient = wdc,
        use_gpu = use_gpu,
        train_data = train_loader,
        val_data = val_loader,
        test_data = test_loader
    )
    start_time = time.time()
    experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
    print("--- %s seconds ---" % (time.time() - start_time))

    #2.C RNN
    exp_name = "exp1_p2_gru"
    gru_params["input_shape"] = input_shape
    gru = GRU1D(gru_params)

    experiment = Experiment(
        network_model = gru,
        experiment_name = results_dir+exp_name,
        num_epochs = num_epochs,
        learning_rate = lr,
        weight_decay_coefficient = wdc,
        use_gpu = use_gpu,
        train_data = train_loader,
        val_data = val_loader,
        test_data = test_loader
    )
    start_time = time.time()
    experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
    print("--- %s seconds ---" % (time.time() - start_time))


    #2.D RNN-attention
    exp_name = "exp1_p2_grusa"
    grusa_params["input_shape"]=input_shape
    grusa = GRU1D(grusa_params)

    experiment = Experiment(
        network_model = grusa,
        experiment_name = results_dir+exp_name,
        num_epochs = num_epochs,
        learning_rate = lr,
        weight_decay_coefficient = wdc,
        use_gpu = use_gpu,
        train_data = train_loader,
        val_data = val_loader,
        test_data = test_loader
    )
    start_time = time.time()
    experiment.run_experiment(test_results="test_results.csv",test_summary="test_summary.csv")
    print("--- %s seconds ---" % (time.time() - start_time))

else:
    print("training and test set need to be the same length")





