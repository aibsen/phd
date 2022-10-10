from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn
from positional_encodings import PositionalEncoding, TimeFiLMEncoding
import numpy as np

class TSTransformer(nn.Module):

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
        self.layer_dict['decoder_in'] = nn.Linear(input_features, d_model,bias=False)
        self.layer_dict['local_encoder_0'] = PositionalEncoding(d_model, dropout,max_len=128)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        self.layer_dict['encoder'] = nn.TransformerEncoder(encoder_layer, nlayers)
        self.layer_dict['local_encoder_1'] = PositionalEncoding(d_model, dropout,max_len=128)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        self.layer_dict['decoder'] = nn.TransformerDecoder(decoder_layer,nlayers)
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

    def forward(self, src, trg) -> Tensor:
        
        src_lens = src[1]
        trg_lens = trg[1]

        src = src[0]
        trg = trg[0]
        # print(src.shape)
        src = self.layer_dict['encoder_in'](src) 
        # print(src.shape)
        src = self.layer_dict['local_encoder_0'](src)
        # print(src.shape)


        trg = self.layer_dict['decoder_in'](trg)
        trg = self.layer_dict['local_encoder_1'](trg)
    #     trg = self.layer_dict['local_encoder_1'](trg[0])
    #     # print(src.shape)
        
        src_mask = self.pad_mask(src,src_lens)
        # print(src_mask.shape)
    #     # src_trg_mask = self.pad_mask(trg, src)
    #     trg_mask = self.pad_mask(trg, trg_lens)
    #     no_peak = self.no_peak_mask(trg)
    #     # print(trg_mask)

    #     # src = src.permute(0,2,1) #bs,length,features
    #     # trg = trg.permute(0,2,1) #bs,length,features
        memory = self.layer_dict['encoder'](src, src_key_padding_mask=src_mask)
        # print(memory.shape)
        out=self.layer_dict['local_decoder'](memory)
        # print(out.shape)
        out = out.view(out.shape[0],out.shape[1]*out.shape[2])
        return(out)
    #     out = self.layer_dict['decoder'](trg, memory, tgt_mask=no_peak,tgt_key_padding_mask=trg_mask)
    #     # , src_trg_mask) not sure if needed

    #     out = self.layer_dict['local_decoder'](out)
        
    #     return out

    def pad_mask(self, q, lens_q):
        # print(q)
        len_q = q.size(1)
        mask = torch.zeros((q.shape[0],q.shape[1]),device=torch.device('cuda'))
        for i,l in enumerate(lens_q):
            mask[i,l:] = 1
        return mask==1 

    # def no_peak_mask(self, s):
    #     sz = s.shape[1]
    #     return torch.triu(torch.ones(sz, sz,device=torch.device('cuda')) * float('-inf'), diagonal=1)

    # # def generate_square_subsequent_mask(sz: int) -> Tensor:
    # #     """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    # #     return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


