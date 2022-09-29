import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn
from positional_encodings import PositionalEncoding, TimeFiLMEncoding
import numpy as np

class TSTransformer(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.2):
        super().__init__()

        self.layer_dict = nn.ModuleDict()
        self.layer_dict['local_encoder_0'] = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.layer_dict['encoder'] = nn.TransformerEncoder(encoder_layer, nlayers)
        
        self.layer_dict['local_encoder_1'] = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.layer_dict['decoder'] = nn.TransformerDecoder(decoder_layer,nlayers)
        self.layer_dict['local_decoder'] = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim()>1:
                try:
                    p.reset_parameters()
                except Exception as e:
                    print(e)
                    nn.init.xavier_uniform_(p)

    def reset_parameters(self) -> None:
        self.init_weights()

    def forward(self, src, trg) -> Tensor:

        src = self.layer_dict['local_encoder_0'](src)
        trg = self.layer_dict['local_encoder_1'](trg)

        src_mask = self.make_pad_mask(src, src)
        src_trg_mask = self.make_pad_mask(trg, src)
        trg_mask = self.make_pad_mask(trg, trg) * \
                   self.make_no_peak_mask(trg, trg)

        memory = self.encoder(src, src_mask)
        out = self.decoder(trg, memory, trg_mask, src_trg_mask)

        out = self.layer_dict['local_decoder'](out)
        
        return out

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        # batch_size x 1 x 1 x len_k
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)
        # batch_size x 1 x len_q x 1
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)
        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

    # def generate_square_subsequent_mask(sz: int) -> Tensor:
    #     """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    #     return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


