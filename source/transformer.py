from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn
from positional_encodings import PositionalEncoding, TimeFiLMEncoding
import numpy as np

class TSTransformerClassifier(nn.Module):

    def __init__(self,
        input_features = 4,
        d_model = 128,
        nhead =1,
        d_hid: int = 150,
        nlayers: int = 1, 
        dropout: float = 0.2,
        max_len: float = 128,
        uneven_t: bool = False,
        num_output_classes = 4):
        super().__init__()

        self.layer_dict = nn.ModuleDict()
        input_features = input_features-1 if uneven_t else input_features
        self.layer_dict['encoder_in'] = nn.Linear(input_features, d_model,bias=False)
        self.layer_dict['positional_encoding'] = PositionalEncoding(d_model, dropout,max_len=max_len, custom_position=uneven_t)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        norm_layer = nn.LayerNorm(d_model)
        self.layer_dict['encoder'] = nn.TransformerEncoder(encoder_layer, nlayers, norm=norm_layer)
        # self.layer_dict['local_decoder'] = nn.Linear(d_model*max_len, num_output_classes)
        self.layer_dict['local_decoder'] = nn.Linear(d_model, d_model)
        self.layer_dict['classifier'] = nn.Linear(d_model, num_output_classes)
        self.layer_dict['dropout'] = nn.Dropout(p=0.2)
        self.uneven_t = uneven_t
        self.max_len = max_len
        self.d_model = d_model
        self.init_weights()

    def init_weights(self) -> None:
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except Exception as e:
                # print(e)
                for p in item.parameters():
                    if p.dim() > 1: 
                        nn.init.xavier_uniform_(p)

    def reset_parameters(self) -> None:
        self.init_weights()

    def forward(self, src):

        # print(type(src))
        # if type(src) is tuple:
        # if self.uneven_t: # seq lengths are different and thus are specified in dataset
        lens = src[1]
        src = src[0]
        #     src = src.permute(0,2,1)
        # pad_mask = self.pad_mask(src,lens)
        #     # x = src[:,:,0:-1] 

        #     # t = src[:,:,-1] #last dimensions represent t
        # else:
        src = src.permute(0,2,1)
        src = src[:,:,0:-1] 
        # pad_mask = torch.zeros((src.shape[0],src.shape[1]), device=torch.device('cuda'))
        pad_mask = self.pad_mask(src,lens)
        x = src
 
        # if self.uneven_t:
        #     x = src[:,:,0:-1] 
        #     t = src[:,:,-1] #last dimensions represent t
        # else: 
        # x = src
        # pad_mask = torch.zeros((src.shape[0],src.shape[1]), device=torch.device('cuda'))

        
        x = self.layer_dict['encoder_in'](x) #input embedding 
        # x = x if not self.uneven_t else (x,t) 
        # x = (x,t) if self.uneven_t else x 
        x = self.layer_dict['positional_encoding'](x) #positional encoding
        x = self.layer_dict['encoder'](x, src_key_padding_mask=pad_mask)
        
        if self.uneven_t:
        #     # print(x.shape)
        #     # print(lens.shape)
            i = lens.unsqueeze(1).long()
        #     # print(x)
            i =  lens.view(-1,1,1).long() - 1
        #     # print(i.shape)
            i = i.repeat(1,1,self.d_model)

        #     # print(i.shape)
        #     torch.set_printoptions(edgeitems=200)
        #     # print(i[0])
        #     # print(x[0][0])
        #     # print(x[0][1])
        #     # print(x[0][2])
        #     # print(x.shape)
        #     # print(i[0])
        #     # print(x[0])
            out = x.gather(1,i) #last representation vector
        # else:
        # out = x[:,-1,:] # if all sequences have the same length, take last element
        # print(out[0])
        # print(out[1])
        # print(out[2])
        # print(out[0][1])
        # print(out[0][2])
        # print(out.shape)
        # print(out.shape)
        # out = out.view(out.shape[0],out.shape[1]*out.shape[2])
        out = out.squeeze()
        out = F.relu(self.layer_dict['local_decoder'](out))
        out = self.layer_dict['dropout'](out)
        out = self.layer_dict['classifier'](out)
        # print(out.shape)
        # out = self.layer_dict['classifier'](out)
        return(out)

    def pad_mask(self, q, lens_q):
        # print(q)
        len_q = q.size(1)
        mask = torch.zeros((q.shape[0],q.shape[1]),device=torch.device('cuda'))
        for i,l in enumerate(lens_q):
            mask[i,l+1:] = 1
        return mask==1 

    # def no_peak_mask(self, s):
    #     sz = s.shape[1]
    #     return torch.triu(torch.ones(sz, sz,device=torch.device('cuda')) * float('-inf'), diagonal=1)

    # # def generate_square_subsequent_mask(sz: int) -> Tensor:
    # #     """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    # #     return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


