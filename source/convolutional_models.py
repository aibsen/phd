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
        self.conv = nn.Conv1d(in_channels=in_channels,kernel_size=ks,out_channels=n_filters)
        self.bn = nn.BatchNorm1d(n_filters)

    def forward(self, x):
        out = x
        out = F.relu(self.bn(self.conv(out)))
        return out

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv.weight, mode = 'fan_in', nonlinearity = 'relu')
        nn.init.constant_(self.conv.bias, 0)
        self.bn.reset_parameters()

class FCNN1D(nn.Module):
    def __init__(self, params=None,regularize=True, seed=1772670):
        super(FCNN1D, self).__init__()

        self.seed = seed
        self.layer_dict = nn.ModuleDict()
        self.params = params
        self.regularize = regularize

        if self.params is not None:
            self.build_module()

    def build_module(self):
        print("")
        print("Building Fully Convolutional Network using input shape", self.params["input_shape"])

        self.layer_dict['conv_block_0'] = Conv1DBlock(in_channels=self.params["input_shape"][0],ks=8,n_filters=128)
        self.layer_dict['conv_block_1'] = Conv1DBlock(in_channels=128,ks=5,n_filters=256)
        self.layer_dict['conv_block_2'] = Conv1DBlock(in_channels=256,ks=3,n_filters=128)
        
        # if self.params["global_pool"] == 'avg':
        #     self.global_pool = torch.nn.AvgPool1d(2)
        # elif self.params["global_pool"] == 'max':
        #     self.global_pool = torch.nn.MaxPool1d(2)

        if self.regularize:
            self.dropout = torch.nn.Dropout(p=0.2)
        self.layer_dict["linear"] = nn.Linear(in_features=128,out_features=self.params['num_output_classes'])


    def forward(self, x):
        out = x
        for i in range(3):
            out = self.layer_dict["conv_block_{}".format(i)](out)
        if self.regularize:
            out = self.dropout(out)
        # out = self.global_pool(out)
        if self.params["global_pool"] == 'avg':
            out = torch.nn.AvgPool1d(out.shape[-1])(out)
        elif self.params["global_pool"] == 'max':
            out = torch.nn.MaxPool1d(out.shape[-1])(out)
        out = out.permute(0,2,1)
        out = out.contiguous().view(out.shape[0], -1)
        out = self.layer_dict["linear"](out)
        return out

    def reset_parameters(self):
        # print("Initializig weights of FCN")
        torch.cuda.manual_seed(self.seed)
        for i,item in enumerate(self.layer_dict.children()):
            try:
                item.reset_parameters()
            except Exception as e:
                print(e)

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
        self.dropout=nn.Dropout(p=0.2)

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
            self.dropout(out)
        return out

    def reset_parameters(self):
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except Exception as e:
                print(e)


class ResNet1D(nn.Module):
    def __init__(self,params=None):
        super(ResNet1D, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.params = params
        if self.params is not None:
            self.build_module()

    def build_module(self):
        print("Building ResNet using input shape", self.params["input_shape"])

        in_channels=self.params["input_shape"][0]
        self.layer_dict["res_block_0"] = ResNet1DBlock(in_channels=in_channels, n_filters=64)
        self.layer_dict["res_block_1"] = ResNet1DBlock(in_channels=64, n_filters=128)
        self.layer_dict["res_block_2"] = ResNet1DBlock(in_channels=128, n_filters=128)
        if self.params["global_pool"] == "max":
            self.global_pool = torch.nn.MaxPool1d(2)
        elif self.params["global_pool"] == "avg":
            self.global_pool = torch.nn.AvgPool1d(2)
        # self.layer_dict['linear'] = nn.Linear(in_features=int(128*self.params["input_shape"][1]/2),
            # out_features=self.params['num_output_classes'])

    def forward(self, x):
        out = x
        for i in range(3):
            out = self.layer_dict["res_block_{}".format(i)](out)
        #permute to do global pooling across filter dimention
        out = out.permute(0,2,1)
        out = self.global_pool(out)
        out = out.contiguous().view(out.shape[0], -1)
        # out = self.layer_dict["linear"](out)
        return out

    def reset_parameters(self):
        print("Initializing weights of ResNet")
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class CNN(nn.Module):
    def __init__(self,params=None):
        super(CNN, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.params = params
        if self.params is not None:
            self.in_channels=self.params["input_shape"][0]
            self.img_size = self.params['input_shape'][1]
            self.n_layers = self.params["n_layers"]
            self.n_filters = self.params["n_filters"]
            self.ks = self.params['kernel_size']
            self.drop_out = self.params['drop_out']
            self.build_module()

    def build_module(self):
        print("Building Vanilla CNN using input shape", self.params["input_shape"])
        in_channels = self.in_channels
        for l in range(self.n_layers):
            self.layer_dict['conv_{}'.format(l)] = nn.Conv2d(in_channels=in_channels,out_channels=self.n_filters,kernel_size=self.ks)
            self.layer_dict['bn_{}'.format(l)] = nn.BatchNorm2d(self.n_filters)
            in_channels=self.n_filters

        #lambda to calculate output size f convolutional blocks
        self.bn = nn.BatchNorm2d(self.n_filters)
        out_f = lambda in_f, k, n: in_f if n == 0 else int(out_f((in_f-(k-1))/2, k, n-1))
        input_linear = self.n_filters*(out_f(self.img_size,self.ks,self.n_layers)**2)
        self.layer_dict["linear_0"] = nn.Linear(in_features=input_linear,out_features=input_linear)
        self.layer_dict["linear_1"] = nn.Linear(in_features=input_linear,out_features=14)


    def forward(self, x):
        out = x
        for i in range(self.n_layers):
            out = F.relu(self.layer_dict['bn_{}'.format(i)](self.layer_dict["conv_{}".format(i)](out)))
            out = nn.MaxPool2d(2)(out)
            out = nn.Dropout(p=self.drop_out)(out)
        out = torch.flatten(out,start_dim=1)
        out = F.relu(self.layer_dict['linear_0'](out))
        out = self.layer_dict['linear_1'](out)
        return out

    def reset_parameters(self):
        print("Initializing weights of Vanilla CNN")
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except Exception as e:
                print(e)

class DMDTShallowCNN(nn.Module):
    def __init__(self,params=None):
        super(CNN, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.params = params
        if self.params is not None:
            self.in_channels=self.params["input_shape"][0]
            self.img_size = self.params['input_shape'][1]
            self.n_layers = self.params["n_layers"]
            self.n_filters = self.params["n_filters"]
            self.ks = self.params['kernel_size']
            self.drop_out = self.params['drop_out']
            self.build_module()

    def build_module(self):
        print("Building Vanilla CNN using input shape", self.params["input_shape"])
        in_channels = self.in_channels
        for l in range(self.n_layers):
            self.layer_dict['conv_{}'.format(l)] = nn.Conv2d(in_channels=in_channels,out_channels=self.n_filters,kernel_size=self.ks)
            self.layer_dict['bn_{}'.format(l)] = nn.BatchNorm2d(self.n_filters)
            in_channels=self.n_filters

        #lambda to calculate output size f convolutional blocks
        self.bn = nn.BatchNorm2d(self.n_filters)
        out_f = lambda in_f, k, n: in_f if n == 0 else int(out_f((in_f-(k-1))/2, k, n-1))
        input_linear = self.n_filters*(out_f(self.img_size,self.ks,self.n_layers)**2)
        self.layer_dict["linear_0"] = nn.Linear(in_features=input_linear,out_features=input_linear)
        self.layer_dict["linear_1"] = nn.Linear(in_features=input_linear,out_features=14)


    def forward(self, x):
        out = x
        for i in range(self.n_layers):
            out = F.relu(self.layer_dict['bn_{}'.format(i)](self.layer_dict["conv_{}".format(i)](out)))
            out = nn.MaxPool2d(2)(out)
            out = nn.Dropout(p=self.drop_out)(out)
        out = torch.flatten(out,start_dim=1)
        out = F.relu(self.layer_dict['linear_0'](out))
        out = self.layer_dict['linear_1'](out)
        return out

    def reset_parameters(self):
        print("Initializing weights of Vanilla CNN")
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except Exception as e:
                print(e)