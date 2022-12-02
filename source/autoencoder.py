import torch
import torch.nn.functional as F
import torch.nn as nn
from positional_encodings import PositionalEncoding
import sys
import numpy as np


class TSAutoencoder(nn.Module):

    def __init__(self,
        input_features = 4, #???
        d_model = 128,
        nhead =8,
        dropout: float = 0.2,
        max_len: float = 128,
        uneven_t: bool = False,
        num_output_classes = 4,
        embedding_layer = None,
        encoder_positional_encoding = None,
        local_decoder = None,
        classifier = None,
        classify = False):
        super().__init__()

        self.layer_dict = nn.ModuleDict()
        input_features = input_features
        #  if not time_dimension else input_features - 1
        
        self.layer_dict['encoder_embedding'] = embedding_layer if embedding_layer\
            is not None else nn.Linear(input_features, d_model,bias=False)
        
        self.layer_dict['encoder_pos'] = encoder_positional_encoding if encoder_positional_encoding\
            is not None else PositionalEncoding(d_model, dropout,max_len=max_len)
            # , custom_position=uneven_t)
        
        self.layer_dict['encoder'] = nn.Sequential(
                    nn.Linear(d_model,d_model),
                    nn.ReLU(),
                    nn.Dropout(p=0.2)
                )

        self.layer_dict['decoder'] = nn.Sequential(
                    nn.Linear(d_model,d_model),
                    nn.ReLU(),
                    nn.Dropout(p=0.2)
                )

        self.layer_dict['local_decoder'] = local_decoder if local_decoder\
            is not None else nn.Linear(d_model,input_features,bias=False)

        self.layer_dict['classifier'] = classifier if classifier\
            is not None else \
                nn.Sequential(
                nn.Linear(d_model,d_model),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(d_model,num_output_classes)
        )

        self.uneven_t = uneven_t
        self.max_len = max_len
        self.d_model = d_model
        self.nheads = nhead
        self.classify = classify

        self.init_weights()

    def init_weights(self) -> None:
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except Exception:
                for p in item.parameters():
                    if p.dim() > 1: 
                        nn.init.xavier_uniform_(p)

    def reset_parameters(self) -> None:
        self.init_weights()


    def freeze_autoencoder(self) -> None:
        for k,v in self.layer_dict.items():
            if k != 'classifier':
                for p in v.parameters():
                    p.requires_grad = False
                    

    def forward(self, seq):
        # if self.uneven_t: # seq lengths are different and thus are specified in dataset
        # print(type(src))
        if type(seq) == list: # seq lengths are different and thus are specified in dataset
            lens = seq[1]
            seq = seq[0]
            seq = seq.permute(0,2,1)
        else:
            seq = seq.permute(0,2,1)
        

        src = seq
        
        src = self.layer_dict['encoder_embedding'](src) #input embedding 
        src = self.layer_dict['encoder_pos'](src) #positional encoding
        memory = self.layer_dict['encoder'](src)#.unsqueeze(0).repeat(src.shape[0]*self.nheads,1,1))*pad_mask)#, src_key_padding_mask=pad_mask)
        # print(memory.shape)
        out_last =  memory[:,-1,:]
        # print(out_last.shape)

        out = self.layer_dict['decoder'](memory)
        # print(out.shape)

        out = self.layer_dict['local_decoder'](out)
        # print(out.shape)
        out = out.permute(0,2,1)
        out = torch.nan_to_num(out)
       
        if self.classify:
            out = self.layer_dict['classifier'](out_last)

        return out
