from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn
from positional_encodings import PositionalEncoding, TimeFiLMEncoding
import numpy as np
from torch.distributions.normal import Normal

import sys


class TSVAE(nn.Module):
    def __init__(self,
        input_features):
        super().__init__()
        
        self.layer_dict = nn.ModuleDict()
        self.input_features = input_features
        self.layer_dict['mu_z'] = nn.Linear(input_features,input_features)
        self.layer_dict['logvar_z'] = nn.Linear(input_features,input_features)


    def forward(self, z):

        mu_z = self.layer_dict['mu_z'](z)
        logvar_z = self.layer_dict['logvar_z'](z)

        s = torch.rand_like(mu_z)
        new_z = mu_z + s * torch.exp(logvar_z)

        return new_z, mu_z, logvar_z

class VaeLoss(nn.Module):
    def __init__(self, 
        k0=1,
        k1=1e-4,
        n_epochs=100,
        cycles=4):

        super().__init__()
        self.device = torch.device('cuda')
        self.reconstruction_loss = torch.nn.MSELoss().to(self.device)
        self.classification_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.k0=k0
        self.k1=k1
        self.n_epochs = n_epochs
        self.n_cycles = cycles
        self.beta = 0

        self.define_schedule()


    def define_schedule(self):
        is_int = self.n_epochs%self.n_cycles
        steps_in_cycle = int(self.n_epochs/self.n_cycles) if is_int==0 else int(self.n_epochs/self.n_cycles+1)
        r = torch.arange(-6,7,13/steps_in_cycle)
        self.beta = torch.sigmoid(r.repeat(self.n_cycles))

    def kl_loss(self, mu_z, logvar_z):
        #for standard normal
        exponential = logvar_z - torch.pow(mu_z,2) - torch.exp(logvar_z) + 1
        result = -0.5 * torch.mean(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(self, tgt, true_tgt,  mu_z, logvar_z, y_pred=None, true_y=None, i=1):
        kl_div = self.kl_loss(mu_z, logvar_z)
        reconstruction_loss = self.reconstruction_loss(tgt, true_tgt)
        if (y_pred is not None):
            classification_loss = self.classification_loss(y_pred, true_y)
            total_loss = (self.k0*reconstruction_loss + self.k1*classification_loss) + self.beta[i]*kl_div
        else:
            total_loss = self.k0*reconstruction_loss + kl_div
        
        return total_loss
 

class TSTransformerVAE(nn.Module):

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
        encoder_positional_encoding = None,
        decoder_positional_encoding = None,
        local_decoder = None,
        classifier = None,
        classify = False,
        k0=1,
        k1=1e-4,
        n_epochs=100,
        cycles=4
        ):
        super().__init__()

        self.layer_dict = nn.ModuleDict()
        input_features = input_features
        #  if not time_dimension else input_features - 1
        
        self.layer_dict['encoder_embedding'] = embedding_layer if embedding_layer\
            is not None else nn.Linear(input_features, d_model,bias=False)
        
        self.layer_dict['encoder_pos'] = encoder_positional_encoding if encoder_positional_encoding\
            is not None else PositionalEncoding(d_model, dropout,max_len=max_len)
            # , custom_position=uneven_t)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        encoder_norm_layer = nn.LayerNorm(d_model)
        
        self.layer_dict['encoder'] = nn.TransformerEncoder(encoder_layer, nlayers, norm=encoder_norm_layer)

        self.layer_dict['vae'] = TSVAE(d_model)

        self.layer_dict['decoder_embedding'] = embedding_layer if embedding_layer\
            is not None else nn.Linear(input_features, d_model,bias=False)
        
        self.layer_dict['decoder_pos'] = decoder_positional_encoding if decoder_positional_encoding\
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
        self.src_mask = self.generate_square_subsequent_mask(self.max_len)
        self.tgt_mask = self.generate_square_subsequent_mask(self.max_len)
        self.memory_mask = self.generate_square_subsequent_mask(self.max_len)

        self.criterion = VaeLoss(k0,k1,n_epochs,cycles)    

        self.init_weights()

    def init_weights(self) -> None:
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except Exception:
                for p in item.parameters():
                    if p.dim() > 1: 
                        nn.init.xavier_uniform_(p)

    def reset_loss(self,n_epochs):
        k0 = self.criterion.k0
        k1 = self.criterion.k1
        n_cycles = self.criterion.n_cycles
        self.criterion = VaeLoss(k0,k1,n_epochs,n_cycles)

    def reset_parameters(self) -> None:
        self.init_weights()

    def freeze_decoder(self) -> None:
        for k,v in self.layer_dict.items():
            if ('decoder' in k) or (k=='vae'):
                print("freezing {}".format(k))
                for p in v.parameters():
                    p.requires_grad = False


    def freeze_autoencoder(self) -> None:
        for k,v in self.layer_dict.items():
            if k != 'classifier':
                print("freezing {}".format(k))
                for p in v.parameters():
                    p.requires_grad = False

    def encode(self,src):
        src = src.permute(0,2,1)
        src = self.layer_dict['encoder_embedding'](src) #input embedding 
        src = self.layer_dict['encoder_pos'](src) #positional encoding
        memory = self.layer_dict['encoder'](src,mask=self.src_mask)
        return memory

    def decode(self,z,tgt,tgt_mask):
        # print(tgt.shape)
        tgt = tgt.permute(0,2,1)
        tgt = self.layer_dict['decoder_embedding'](tgt)
        tgt = self.layer_dict['decoder_pos'](tgt)
        # print(tgt.shape)
        out = self.layer_dict['decoder'](tgt=tgt, memory=z,\
            tgt_mask=tgt_mask, memory_mask=self.memory_mask)# memory_key_padding_mask=memory_pad_mask)#,\
            # tgt_key_padding_mask=pad_mask)#, memory_pad_mask)
        return out

    def local_decode(self, out):
        out = self.layer_dict['local_decoder'](out)
        out = out.permute(0,2,1)
        out = torch.nan_to_num(out)
        # print(out.shape)
        return out


    def forward(self, src,tgt=None,tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = self.tgt_mask
      
        if tgt is None and type(src==list): #coming from classification experiment, src is x,lenx
            src = src[0] #why is it so convoluted? cos I need to submit next week thats why
            tgt = src.new_zeros(src.shape)
            tgt[:,:,1:] = src[:,:,:-1]
            tgt[:,:,0] = -1

        memory = self.encode(src)
        z, mu_z, logvar_z = self.layer_dict['vae'](memory)
       
        out = self.decode(z, tgt, tgt_mask)

        out_last = z[:,-1,:]
        label = self.layer_dict['classifier'](out_last)
        if self.classify:
            return label

        return self.local_decode(out), label, mu_z, logvar_z

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = torch.triu(torch.ones(sz, sz) * float(1), diagonal=1).cuda()
        return mask==float(1)
        # return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).cuda()
