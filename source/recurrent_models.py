import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

class SelfAttention1D(nn.Module):

    def __init__(self, params, da=50, r=1):
        super(SelfAttention1D, self).__init__()
        self.params = params
        if self.params["r"]:
            self.r=self.params["r"]
        else:
            self.r=r

        if self.params["da"]:
            self.da=self.params["da"]
        else:
            self.da = da

        self.layer_dict = nn.ModuleDict()
        if self.params is not None:
            self.build_module()

    def build_module(self):
        self.layer_dict['weighted_h'] = nn.Linear(self.params["hidden_size"], self.da ,bias=False)
        self.layer_dict['e'] = nn.Linear(self.da, self.r,bias=False)

    def forward(self, h):
        # print(h.shape)
        weighted_h = self.layer_dict["weighted_h"](h)
        e = self.layer_dict["e"](torch.tanh(weighted_h))
        a = F.softmax(e, dim=1)
        a = a.permute(0,2,1)
        # print(a.shape)
        context = torch.bmm(a,h)
        # print(context.shape)
        return context

    def reset_parameters(self):
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class GRU1D(nn.Module):
    def __init__(self, params):
        super(GRU1D, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.params = params

        if self.params is not None:
            self.build_module()

    def build_module(self):
        print("Building basic block of GRU ensemble using input shape", self.params["input_shape"])
        print(self.params)
        self.layer_dict["gru_0"] = nn.GRU(input_size=self.params["input_shape"][0],hidden_size = self.params["hidden_size"],batch_first=True)
        self.layer_dict['dropout_0'] = torch.nn.Dropout(p=0.2)
        self.layer_dict['bn_0'] = nn.BatchNorm1d(self.params["hidden_size"])
        self.layer_dict["gru_1"] = nn.GRU(self.params["hidden_size"], self.params["hidden_size"], batch_first=True)
        self.layer_dict['dropout_1'] = torch.nn.Dropout(p=0.2)
        self.layer_dict['bn_1'] = nn.BatchNorm1d(self.params['hidden_size'])
        self.layer_dict['dropout'] = torch.nn.Dropout(p=0.2)
        self.layer_dict["self_attention"] = SelfAttention1D(self.params)
        if self.params["r"] > 1:
            self.layer_dict['linear'] = nn.Linear(in_features=self.params['hidden_size']*self.params["r"],out_features=self.params['num_output_classes'])
        else:
            self.layer_dict['linear'] = nn.Linear(in_features=self.params['hidden_size'],out_features=self.params['num_output_classes'])

    def forward(self, x):
        out = x.permute(0,2,1)
        for i in range(2):
            out,h = self.layer_dict["gru_{}".format(i)](out)
            out = self.layer_dict["dropout_{}".format(i)](out)
            out = out.permute(0,2,1)
            out = self.layer_dict["bn_{}".format(i)](out)
            out = out.permute(0,2,1)
        out = self.layer_dict["dropout"](out)
        if self.params["attention"] == "self_attention":
            out = self.layer_dict["self_attention"](out)
            if self.params["r"]>1:
                out = out = out.contiguous().view(out.shape[0], -1)
            out = self.layer_dict["linear"](out)
            return out.squeeze()
        elif self.params["attention"] == "no_attention":
            out = self.layer_dict["linear"](out)
            return out[:,-1,:]

    def reset_parameters(self):
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass
