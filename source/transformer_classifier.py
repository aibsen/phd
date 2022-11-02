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
        dropout: float = 0.2):
        super().__init__()

        self.layer_dict = nn.ModuleDict()
        self.layer_dict['encoder_in'] = nn.Linear(input_features, d_model,bias=False)
        self.layer_dict['local_encoder_0'] = PositionalEncoding(d_model, dropout,max_len=128)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        self.layer_dict['encoder'] = nn.TransformerEncoder(encoder_layer, nlayers)
        self.layer_dict['local_decoder'] = nn.Linear(d_model, 6)

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

    def forward(self, src) -> Tensor:
        
        src_lens = src[1]
        src = src[0]
        src = self.layer_dict['encoder_in'](src) 

        src_mask = self.pad_mask(src,src_lens)
        code = self.layer_dict['encoder'](src, src_key_padding_mask=src_mask)
        out=self.layer_dict['local_decoder'](code)
        out = out.view(out.shape[0],out.shape[1]*out.shape[2])
        return(out)

    def pad_mask(self, q, lens_q):
        len_q = q.size(1)
        mask = torch.zeros((q.shape[0],q.shape[1]),device=torch.device('cuda'))
        for i,l in enumerate(lens_q):
            mask[i,l:] = 1
        return mask==1 
