from torch import nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import tqdm
import os
import numpy as np

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, ks=3, n_filters=128):
        super(Conv1DBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['conv'] = nn.Conv1d(in_channels=in_channels,kernel_size=ks,out_channels=n_filters)
        self.layer_dict['bn'] = nn.BatchNorm1d(n_filters)

    def forward(self, x):
        out = x
        out = F.relu(
            self.layer_dict['bn'](
                self.layer_dict['conv'](out)))
        return out


class FCNN1D(nn.Module):
    def __init__(self, params=None):
        super(FCNN1D, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.params = params

        if self.params is not None:
            self.build_module()

    def build_module(self):
        print("Building Fully Convolutional Network using input shape", self.params["input_shape"])
        print(self.params)
        self.layer_dict['conv_block_0'] = Conv1DBlock(in_channels=self.params["input_shape"][0],ks=8,n_filters=128)
        self.layer_dict['conv_block_1'] = Conv1DBlock(in_channels=128,ks=5,n_filters=256)
        self.layer_dict['conv_block_2'] = Conv1DBlock(in_channels=256,ks=3,n_filters=128)
        
        if self.params["global_pool"] == 'avg':
            self.layer_dict['global_pool'] = torch.nn.AvgPool1d(2)
        elif self.params["global_pool"] == 'max':
            self.layer_dict['global_pool'] = torch.nn.MaxPool1d(2)

        if self.params["regularize"]:
            self.layer_dict['dropout'] = torch.nn.Dropout(p=0.2)
        self.layer_dict["linear"] = nn.Linear(in_features=(self.params["input_shape"][1]-13)*64,out_features=self.params['num_output_classes'])
    
    def forward(self, x):
        out = x
        for i in range(3):
            out = self.layer_dict["conv_block_{}".format(i)](out)

        out = out.permute(0,2,1)
        if self.params["regularize"]:
            out = self.layer_dict["dropout"](out)
        out = self.layer_dict['global_pool'](out)
        out = out.contiguous().view(out.shape[0], -1)
        out = self.layer_dict["linear"](out)
        return out

    def reset_parameters(self):
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass


class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels,n_filters=64,regularize=True):
        super(ResNet1DBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.regularize=regularize
        self.layer_dict['conv_block_0'] = Conv1DBlock(in_channels=in_channels,ks=8,n_filters=n_filters)
        self.layer_dict['conv_block_1'] = Conv1DBlock(in_channels=n_filters,ks=5,n_filters=n_filters)
        self.layer_dict['conv_block_2'] = Conv1DBlock(in_channels=n_filters,ks=3,n_filters=n_filters)
        self.layer_dict['expand_res_channels'] = nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size =1)
        self.layer_dict['bn_res']=nn.BatchNorm1d(n_filters)
        self.layer_dict['dropout']=nn.Dropout(p=0.2)

    def forward(self, x):
        out = x
        res = out
        in_length = out.shape[2]
        for i in range(3):
            out = self.layer_dict['conv_block_{}'.format(i)](out)
            out_length = out.shape[2]
            l_diff = in_length-out_length
            padding = nn.ConstantPad1d((0,l_diff),0)
            out = padding(out)
        if res.shape[1]!=out.shape[1]:
            res = self.layer_dict["expand_res_channels"](res)
            res=self.layer_dict["bn_res"](out)
        out = F.relu(out+res)
        if self.regularize:
            self.layer_dict["dropout"](out)
        return out


class ResNet1D(nn.Module):
    def __init__(self,params=None):
        super(ResNet1D, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.params = params
        if self.params is not None:
            self.build_module()

    def build_module(self):
        print("Building ResNet using input shape", self.params["input_shape"])
        print(self.params)
        in_channels=self.params["input_shape"][0]
        self.layer_dict["res_block_0"] = ResNet1DBlock(in_channels=in_channels, n_filters=64)
        self.layer_dict["res_block_1"] = ResNet1DBlock(in_channels=64, n_filters=128)
        self.layer_dict["res_block_2"] = ResNet1DBlock(in_channels=128, n_filters=128)
        if self.params["global_pool"] == "max":
            self.layer_dict['global_pool'] = torch.nn.MaxPool1d(2)
        elif self.params["global_pool"] == "avg":
            self.layer_dict['global_pool'] = torch.nn.AvgPool1d(2)
        self.layer_dict['linear'] = nn.Linear(in_features=int(128*self.params["input_shape"][1]/2),
            out_features=self.params['num_output_classes'])

    def forward(self, x):
        out = x
        for i in range(3):
            out = self.layer_dict["res_block_{}".format(i)](out)
        #permute to do global pooling across filter dimention
        out = out.permute(0,2,1)
        out = self.layer_dict['global_pool'](out)
        out = out.contiguous().view(out.shape[0], -1)
        out = self.layer_dict["linear"](out)
        return out

    def reset_parameters(self):
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass