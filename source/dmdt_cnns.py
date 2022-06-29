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

class VanillaCNN(nn.Module):
    def __init__(self,params=None):
        super(VanillaCNN, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.params = params
        if self.params is not None:
            self.in_channels=self.params["input_shape"][0]
            self.img_size = self.params['input_shape'][1]
            self.num_output_classes = self.params['num_output_classes']

            self.n_conv_layers = 1 if not "n_conv_layers" in params else self.params["n_conv_layers"]
            self.n_filters = 32 if not "n_filters" in params else self.params["n_filters"]
            self.ks = 3 if not "kernel_size" in params else self.params['kernel_size']
            self.pool_size = 2 if not "pool_size" in params else self.params['pool_size']
            self.drop_out_conv = 0.1 if not "drop_out_conv" in params else self.params['drop_out_conv']
            self.drop_out_linear = 0.25 if not "drop_out_linear" in params else self.params['drop_out_linear']
            self.out_linear = 128 if not "out_linear" in params else self.params['out_linear']

            print("Building Vanilla CNN using input shape", self.params["input_shape"])
            in_channels = self.in_channels
            for l in range(self.n_conv_layers):
                if type(self.ks) is int:
                    self.layer_dict['conv_{}'.format(l)] = nn.Conv2d(in_channels=in_channels,out_channels=self.n_filters,kernel_size=self.ks)
                elif type(self.ks) is dict:
                    self.layer_dict['conv_{}'.format(l)] = nn.Conv2d(in_channels=in_channels,out_channels=self.n_filters,kernel_size=self.ks[str(l)])

                # self.layer_dict['bn_{}'.format(l)] = nn.BatchNorm2d(self.n_filters)
                in_channels=self.n_filters
            # out_f = lambda in_f, k, n: in_f if n == 0 else int(out_f((in_f-(k-1))/self.pool_size, k, n-1)) 
            # input_linear = self.n_filters*(out_f(self.img_size,self.ks,self.n_conv_layers)**2)
            if type(self.ks) is int:
                #lambda to calculate output size f convolutional blocks
                out_f = lambda in_f, k, n: in_f if n == 1 else int(out_f((in_f-(k-1))/self.pool_size, k, n-1)) #up to second to last layer
                input_linear = out_f(self.img_size,self.ks,self.n_conv_layers)
                input_linear = input_linear-(self.ks-1) #last convolution is not followed by pooling
                input_linear = self.n_filters*(input_linear**2)
            elif type(self.ks) is dict:
                out_f = lambda in_f, k, n: in_f if n == 1 else int(out_f((in_f-(k[str(self.n_conv_layers-n)]-1))/self.pool_size, k, n-1)) #up to second to last layer
                input_linear = out_f(self.img_size,self.ks,self.n_conv_layers)
                input_linear = input_linear-(self.ks[str(self.n_conv_layers -1)]-1) #last convolution is not followed by pooling
                input_linear = self.n_filters*(input_linear**2)
            
            print(input_linear)
            # input_linear = self.n_filters*(out_f(out_first_conv,self.ks,self.n_conv_layers)**2)
            self.layer_dict["linear_0"] = nn.Linear(in_features=input_linear,out_features=self.out_linear)
            self.layer_dict["linear_1"] = nn.Linear(in_features=self.out_linear,out_features=self.num_output_classes)
            self.do_conv =  nn.Dropout(p=self.drop_out_conv)
            self.do_linear =  nn.Dropout(p=self.drop_out_linear)

    def forward(self, x):
        out = x
        for i in range(self.n_conv_layers-1):
            out = F.relu(self.layer_dict["conv_{}".format(i)](out))
            # out = F.relu(self.layer_dict['bn_{}'.format(i)](self.layer_dict["conv_{}".format(i)](out)))
            out = nn.MaxPool2d(self.pool_size)(out)
            out = self.do_conv(out)
        #last conv not followed by pooling
        out = F.relu(self.layer_dict["conv_{}".format(self.n_conv_layers-1)](out))
        # out = F.relu(self.layer_dict['bn_{}'.format(self.n_conv_layers-1)](self.layer_dict["conv_{}".format(self.n_conv_layers-1)](out)))
        out = self.do_conv(out)

        out = torch.flatten(out,start_dim=1)
        out = F.relu(self.layer_dict['linear_0'](out))
        out = self.do_linear(out)
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
        super(DMDTShallowCNN, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.params = params
        if self.params is not None:
            self.in_channels=self.params["input_shape"][0]
            self.img_size = self.params['input_shape'][1]
            self.n_filters = self.params["n_filters"]
            self.ks = self.params['kernel_size']

            print("Building Shallow CNN using input shape", self.params["input_shape"])
            self.layer_dict['conv'] = nn.Conv2d(in_channels=self.in_channels,out_channels=self.n_filters,kernel_size=self.ks)
            #lambda to calculate output size f convolutional blocks
            out_f = int((self.img_size-(self.ks-1)))
            # print(out_f)
            input_linear = self.n_filters*((out_f)**2)
            # print(input_linear)
            self.layer_dict["linear_0"] = nn.Linear(in_features=input_linear,out_features=128)
            self.layer_dict["linear_1"] = nn.Linear(in_features=128,out_features=14)
            self.do_conv = nn.Dropout(p=0.1)
            self.do_linear = nn.Dropout(p=0.25)

    def forward(self, x):
        out = x
        out = F.relu(self.layer_dict["conv"](out))
        out = self.do_conv(out)
        out = torch.flatten(out,start_dim=1)
        out = F.relu(self.layer_dict['linear_0'](out))
        out = self.do_linear(out)
        out = self.layer_dict['linear_1'](out)
        return out

    def reset_parameters(self):
        print("Initializing weights of Shallow CNN")
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except Exception as e:
                print(e)


class DMDTCNN(nn.Module):
    def __init__(self,params=None):
        super(DMDTCNN, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.params = params
        if self.params is not None:
            self.in_channels=self.params["input_shape"][0]
            self.img_size = self.params['input_shape'][1]
            # self.n_layers = self.params["n_layers"]
            self.n_filters = self.params["n_filters"]
            self.ks = self.params['kernel_size']

            print("Building DMDTCNN using input shape", self.params["input_shape"])
            in_channels = self.in_channels
            self.layer_dict['conv_0'] = nn.Conv2d(in_channels=in_channels,out_channels=self.n_filters,kernel_size=self.ks)
            self.layer_dict['conv_1'] = nn.Conv2d(in_channels=self.n_filters,out_channels=self.n_filters*2,kernel_size=self.ks)
            self.layer_dict['conv_2'] = nn.Conv2d(in_channels=self.n_filters*2,out_channels=self.n_filters*2*2,kernel_size=self.ks)

            #lambda to calculate output size f convolutional blocks

            
            input_linear = (self.n_filters*2*2)*int((self.img_size - (self.ks -1))/2 -2*(self.ks-1))**2 #after 1 conv, 1 maxpool and 2 conv
            output_linear = int(self.n_filters*2*2*2)
            # print(input_linear)
            # print(output_linear)
            self.layer_dict["linear_0"] = nn.Linear(in_features=input_linear,out_features=output_linear)
            self.layer_dict["linear_1"] = nn.Linear(in_features=output_linear,out_features=output_linear)
            self.layer_dict["linear_2"] = nn.Linear(in_features=output_linear,out_features=14)
            self.do_conv = nn.Dropout(p=0.1)
            self.do_linear = nn.Dropout(p=0.5)

    def forward(self, x):
        out = x
        # print(out.shape)

        out = F.relu(self.layer_dict["conv_0"](out))
        # print(out.shape)

        out = nn.MaxPool2d(2)(out)
        # print(out.shape)

        out = self.do_conv(out)
        out = F.relu(self.layer_dict["conv_1"](out))
        # print(out.shape)

        out = F.relu(self.layer_dict["conv_2"](out))
        # print(out.shape)
        out = torch.flatten(out,start_dim=1)
        # print(out.shape)
        out = F.relu(self.layer_dict['linear_0'](out))
        # print(out.shape)
        out = self.do_linear(out)
        out = F.relu(self.layer_dict['linear_1'](out))
        # print(out.shape)

        out = self.layer_dict['linear_2'](out)
        return out

    def reset_parameters(self):
        print("Initializing weights of DMDTCNN")
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except Exception as e:
                print(e)