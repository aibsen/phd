from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn
from positional_encodings import PositionalEncoding, TimeFiLMEncoding
import numpy as np

import sys

class TSTransformerAutoencoder(nn.Module):

    def __init__(self,
        input_features = 4, #???
        d_model = 128,
        nhead =8,
        d_hid: int = 150,
        nlayers: int = 1, 
        dropout: float = 0.2,
        max_len: float = 128,
        uneven_t: bool = False,
        num_output_classes = 4,
        embedding_layer = None,
        positional_encoding = None,
        local_decoder = None,
        reduction = 'last',
        classifier = None,
        classify = False):
        super().__init__()

        self.layer_dict = nn.ModuleDict()
        input_features = input_features
        #  if not time_dimension else input_features - 1
        
        self.layer_dict['encoder_embedding'] = embedding_layer if embedding_layer\
            is not None else nn.Linear(input_features, d_model,bias=False)
        
        self.layer_dict['encoder_pos'] = positional_encoding if positional_encoding\
            is not None else PositionalEncoding(d_model, dropout,max_len=max_len)
            # , custom_position=uneven_t)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        encoder_norm_layer = nn.LayerNorm(d_model)
        
        self.layer_dict['encoder'] = nn.TransformerEncoder(encoder_layer, nlayers, norm=encoder_norm_layer)

        self.layer_dict['decoder_embedding'] = embedding_layer if embedding_layer\
            is not None else nn.Linear(input_features, d_model,bias=False)
        
        self.layer_dict['decoder_pos'] = positional_encoding if positional_encoding\
            is not None else PositionalEncoding(d_model, dropout,max_len=max_len)
            # , custom_position=uneven_t)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        decoder_norm_layer = nn.LayerNorm(d_model)

        self.layer_dict['decoder'] = nn.TransformerDecoder(decoder_layer, nlayers, norm=decoder_norm_layer)
        self.layer_dict['local_decoder'] = local_decoder if local_decoder\
            is not None else nn.Linear(d_model,input_features,bias=False)

        self.layer_dict['classifier'] = classifier if classifier\
            is not None else \
                nn.Sequential(
                nn.Linear(d_model,d_model),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(d_model,num_output_classes)
        )

        self.uneven_t = uneven_t
        self.max_len = max_len
        self.d_model = d_model
        self.nheads = nhead
        self.classify = classify
        self.tgt_mask = self.generate_square_subsequent_mask(self.max_len)
        self.memory_mask = self.generate_square_subsequent_mask(self.max_len)
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
        tgt = seq
        
        src = self.layer_dict['encoder_embedding'](src) #input embedding 
        src = self.layer_dict['encoder_pos'](src) #positional encoding
        memory = self.layer_dict['encoder'](src,mask=self.tgt_mask)#.unsqueeze(0).repeat(src.shape[0]*self.nheads,1,1))*pad_mask)#, src_key_padding_mask=pad_mask)
        tgt = self.layer_dict['decoder_embedding'](tgt)
        tgt = self.layer_dict['decoder_pos'](tgt)
        out = self.layer_dict['decoder'](tgt=tgt, memory=memory,\
            tgt_mask=self.tgt_mask, memory_mask=self.memory_mask)# memory_key_padding_mask=memory_pad_mask)#,\
            # tgt_key_padding_mask=pad_mask)#, memory_pad_mask)
        
        out = self.layer_dict['local_decoder'](out)
        out = out.permute(0,2,1)
        out = torch.nan_to_num(out)
       
        if self.classify:
            out_last = memory[:,-1,:]
            out = self.layer_dict['classifier'](out_last)

        return out

    def pad_mask(self, q, lens_q):
        len_q = q.size(1)
        mask = torch.zeros((q.shape[0],q.shape[1]),device=torch.device('cuda'))
        for i,l in enumerate(lens_q):
            # mask[i,l:] = 1
            mask[i,0:-l] = 1
            
        return mask==1 

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = torch.triu(torch.ones(sz, sz) * float(1), diagonal=1).cuda()
        return mask==float(1)
        # return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).cuda()
