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
# from recurrent_models import GRU1D
from convolutional_models import FCNN1D, ResNet1D
from experiment import Experiment
from plot_utils import *
from torchvision import transforms
from seeded_experiment import SeededExperiment
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix


results_dir = "../../results/"
results_file = "../../results/sa-check/result_outputs/validation_results.csv"
interpolated_dataset_filename = "../../data/training/linearly_interpolated/sn_0000000_m.h5"

lc_length = 128
num_epochs = 30
seed = 77777
torch.manual_seed(seed=seed)
use_gpu = True
lr = 1e-03
wdc = 1e-03
batch_size = 64


#load dataset
interpolated_dataset = LCs(lc_length, interpolated_dataset_filename)
dataset_length = len(interpolated_dataset)
# print(dataset_length)

print(interpolated_dataset.targets)
#split into train/validation/test, validation/test will be ~ .4
# val_length = int(dataset_length/4)
# val_length = int(dataset_length/3)
# train_length = dataset_length -val_length
# train_dataset, val_dataset = torch.utils.data.random_split(interpolated_dataset, [train_length, val_length])
# input_shape = train_dataset[0][0].shape
# # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# lc_0 = interpolated_dataset[26729]
# lc_x = lc_0[0][None,:]


# class SelfAttention1DVis(nn.Module):

#     def __init__(self, params, da=50, r=1):
#         super(SelfAttention1DVis, self).__init__()
#         self.params = params
#         if self.params["r"]:
#             self.r=self.params["r"]
#         else:
#             self.r=r

#         if self.params["da"]:
#             self.da=self.params["da"]
#         else:
#             self.da = da

#         self.layer_dict = nn.ModuleDict()
#         if self.params is not None:
#             self.build_module()

#     def build_module(self):
#         self.layer_dict['weighted_h'] = nn.Linear(self.params["hidden_size"], self.da ,bias=False)
#         self.layer_dict['e'] = nn.Linear(self.da, self.r,bias=False)

#     def forward(self, h):
#         # print(h.shape)
#         weighted_h = self.layer_dict["weighted_h"](h)
#         # print(weighted_h.shape)
#         e = self.layer_dict["e"](torch.tanh(weighted_h))
#         # print(e.shape)
#         a = F.softmax(e, dim=1)
#         a = a.permute(0,2,1)
#         # print(a.shape)
#         context = torch.bmm(a,h)
#         # print(context.shape)
#         return context, a, h 

#     def reset_parameters(self):
#         for item in self.layer_dict.children():
#             try:
#                 item.reset_parameters()
#             except:
#                 pass


# class GRU1D(nn.Module):
#     def __init__(self, params):
#         super(GRU1D, self).__init__()
#         self.layer_dict = nn.ModuleDict()
#         self.params = params

#         if self.params is not None:
#             self.build_module()

#     def build_module(self):
#         print("Building basic block of GRU ensemble using input shape", self.params["input_shape"])
#         print(self.params)
#         self.layer_dict["gru_0"] = nn.GRU(input_size=self.params["input_shape"][0],hidden_size = self.params["hidden_size"],batch_first=True)
#         self.layer_dict['dropout_0'] = torch.nn.Dropout(p=0.2)
#         self.layer_dict['bn_0'] = nn.BatchNorm1d(self.params["hidden_size"])
#         self.layer_dict["gru_1"] = nn.GRU(self.params["hidden_size"], self.params["hidden_size"], batch_first=True)
#         self.layer_dict['dropout_1'] = torch.nn.Dropout(p=0.2)
#         self.layer_dict['bn_1'] = nn.BatchNorm1d(self.params['hidden_size'])
#         self.layer_dict['dropout'] = torch.nn.Dropout(p=0.2)
#         self.layer_dict["self_attention"] = SelfAttention1DVis(self.params)
#         if self.params["r"] > 1 and self.params["attention"]=="self_attention":
#             self.layer_dict['linear'] = nn.Linear(in_features=self.params['hidden_size']*self.params["r"],out_features=self.params['num_output_classes'])
#         else:
#             self.layer_dict['linear'] = nn.Linear(in_features=self.params['hidden_size'],out_features=self.params['num_output_classes'])

#     def forward(self, x):
#         out = x.permute(0,2,1)
#         print(out.shape)
#         for i in range(2):
#             out,h = self.layer_dict["gru_{}".format(i)](out)
#             print(out.shape)
#             out = self.layer_dict["dropout_{}".format(i)](out)
#             out = out.permute(0,2,1)
#             out = self.layer_dict["bn_{}".format(i)](out)
#             out = out.permute(0,2,1)
#         out = self.layer_dict["dropout"](out)
#         if self.params["attention"] == "self_attention":
#             out, a, h = self.layer_dict["self_attention"](out)
#             if self.params["r"]>1:
#                 out = out.contiguous().view(out.shape[0], -1)
#             out = self.layer_dict["linear"](out)
#             return out.squeeze(), a, h

#     def reset_parameters(self):
#         for item in self.layer_dict.children():
#             try:
#                 item.reset_parameters()
#             except:
#                 pass



# exp_name = "sa-check"
# # #define network parameters

# grusa_params = {
#     "input_shape": (4,128),
#     "num_output_classes" : 4,
#     "hidden_size":10,
#     "batch_size":batch_size,
#     "attention":"self_attention",
#     "da":10,
#     "r":128
#     }

# model = GRU1D(grusa_params)


# # experiment = Experiment(
# #                 network_model = model,
# #                 experiment_name = exp_name,
# #                 num_epochs = 1,
# #                 learning_rate = lr,
# #                 weight_decay_coefficient = wdc,
# #                 use_gpu = True,
# #                 batch_size = batch_size,
# #                 num_output_classes= 4,
# #                 train_data = train_dataset,
# #                 val_data = val_dataset,
# #                 verbose = True
# #             )

# # start_time = time.time()
# # experiment.run_experiment()
# # print("--- %s seconds ---" % (time.time() - start_time))


# # items = set(interpolated_dataset.get_all_labels().cpu().numpy())

# filename = "sa-check/saved_models/train_model_f1_score_0"
# state = torch.load(f=filename)
# model.load_state_dict(state_dict=state['network'])
# device = torch.device("cuda")
# model.to(device)
# model.eval()
# with torch.no_grad():
#     out, a, h= model.forward(lc_x)
#     a = a.squeeze().cpu()
#     h = h.squeeze().cpu()
#     # h = h.permute(1,0)
#     # print(h.shape)
#     # print(a.shape)
#     c = torch.mm(a,h)
#     # print(c.shape)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# im = ax.imshow(h,interpolation='nearest')
# cbar = ax.figure.colorbar(im, ax=ax)
# plt.show()

# def plot_cm(ax,true_targets, predictions, normalized=True,colormap=None):
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     cm=confusion_matrix(true_targets,predictions)
#     if normalized:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     if colormap is not None:
#         im = ax.imshow(cm, interpolation= 'nearest', cmap=colormap)
#     else : 
#         im = ax.imshow(cm, interpolation= 'nearest', cmap=plt.cm.PuBu)
#     names = ["snIa"," ","snIb/c"," ","snIIn"," ","snIIP"]
#     # namesy = ["snIa"," ","snIb/c"," ","snIIn"," ","snIIP"]
#     # namesx = ["snIa","snIb/c","snIIn","snIIP"]
#     fmt ='.2f'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             if normalized:
#                 ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#             else :
#                 ax.text(j, i, format(cm[i, j]),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     ax.set_xticklabels([''] + names)
#     ax.set_yticklabels([''] + names)
#     ax.set_xlabel("predicted class")
#     ax.set_ylabel("true class")