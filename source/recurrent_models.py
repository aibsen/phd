import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

class AdditiveAttention1D(nn.Module):

    def __init__(self, params, da=50, r=1):
        super(AdditiveAttention1D, self).__init__()
        self.params = params

        self.r= r if 'r' not in self.params.keys() else self.params['r']  
        self.da = da if 'da' not in self.params.keys() else self.params['da']

        self.linear1 = nn.Linear(self.params["hidden_size"], self.da ,bias=False)
        self.linear2 = nn.Linear(self.da, self.r,bias=False)

    def forward(self, h):
        out = self.linear1(h)
        out = self.linear2(torch.tanh(out))

        a = F.softmax(out, dim=1)
        a = a.permute(0,2,1)
        context = torch.bmm(a,h)
        return context

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)


class GRU1D(nn.Module):
    def __init__(self, params, n_layers=2, dropout=0.2):
        super(GRU1D, self).__init__()

        self.params = params
        print("Building basic block of GRU ensemble using input shape", self.params["input_shape"])
        self.layer_dict = nn.ModuleDict()
        
        self.h_in = self.params['input_shape'][0]
        self.sequence_length = self.params['input_shape'][1]
        self.h_out = self.params["hidden_size"]
        self.n_layers = n_layers if 'n_layers' not in self.params.keys() else self.params['n_layers']
        dropout = dropout if 'dropout' not in self.params.keys() else self.params['dropout']

        self.layer_dict['gru'] = nn.GRU(input_size=self.h_in, hidden_size = self.h_out, batch_first=True, dropout = dropout, num_layers = self.n_layers)
        self.layer_dict['bn'] = nn.BatchNorm1d(self.h_out)
        self.dropout = torch.nn.Dropout(dropout)

        self.r = 1 
        
        if 'attention' in self.params.keys() and self.params['attention']:
            self.r = 1 if 'r' not in self.params.keys() else self.params['r'] 
            self.layer_dict['attn'] = AdditiveAttention1D(self.params, r=self.r)

        self.layer_dict['linear'] = nn.Linear(in_features=self.h_out*self.r,out_features=self.params['num_output_classes'])



    def forward(self, x):
        out = x[0].permute(0,2,1)
        lens = x[1]
        out = nn.utils.rnn.pack_padded_sequence(out,lens,batch_first=True,enforce_sorted=False)
        out, h = self.layer_dict['gru'](out)
        # print(out.shape)
        out, lens = nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=self.sequence_length)
        out = out.permute(0,2,1)
        out = self.layer_dict['bn'](out)
        out = out.permute(0,2,1)
        # print(out.shape)
        if 'attn' in self.layer_dict.keys(): 
            out = self.layer_dict['attn'](out)
            # print(out.shape)
            out = out.view(out.shape[0],self.r*self.h_out)
            # print(out.shape)
            out = self.dropout(out)
        else:
            # print(out.shape)
            idx = lens.max()
            # print(idx)
            out = out[:,idx-1,:]
            out = self.dropout(out)

        # print(out.shape)
        out = self.layer_dict['linear'](out)
        # print(out.shape)
        return out

    def reset_gru_layer(self):

        gru_attrs = self.layer_dict['gru']._flat_weights
        gru_attr_names = self.layer_dict['gru']._flat_weights_names

        for item,name in zip(gru_attrs, gru_attr_names):
            if 'weight'in name:
                for idx in np.arange(3):
                    if 'ih' in name:
                        nn.init.xavier_uniform_(item[idx*self.h_out:(idx+1)*self.h_out])
                    elif 'hh' in name:
                        nn.init.orthogonal_(item[idx*self.h_out:(idx+1)*self.h_out])
            elif 'bias' in name:
                nn.init.constant_(item, 0)


    def reset_parameters(self):
        print("Initializing weights of GRU")
        self.reset_gru_layer()
        self.layer_dict['bn'].reset_parameters()
        if 'attn' in self.layer_dict.keys(): 
            self.layer_dict['attn'].reset_parameters()
        self.layer_dict['linear'].reset_parameters()
                
