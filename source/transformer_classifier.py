from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn
from positional_encodings import PositionalEncoding, TimeFiLMEncoding
import numpy as np

class TSTransformerClassifier(nn.Module):

    def __init__(self,
        input_features = 4, #???
        d_model = 128,
        nhead =1,
        d_hid: int = 150,
        nlayers: int = 1, 
        dropout: float = 0.2,
        max_len: float = 128,
        uneven_t: bool = False,
        time_dimension = False,
        num_output_classes = 4,
        embedding_layer = None,
        positional_encoding = None,
        local_decoder = None,
        reduction = 'last',
        classifier = None):
        super().__init__()

        self.layer_dict = nn.ModuleDict()
        input_features = input_features
        #  if not time_dimension else input_features - 1
        
        self.layer_dict['embedding'] = embedding_layer if embedding_layer\
            is not None else nn.Linear(input_features, d_model,bias=False)
        
        self.layer_dict['pos'] = positional_encoding if positional_encoding\
            is not None else PositionalEncoding(d_model, dropout,max_len=max_len)
            # , custom_position=uneven_t)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        norm_layer = nn.LayerNorm(d_model)
        
        self.layer_dict['encoder'] = nn.TransformerEncoder(encoder_layer, nlayers, norm=norm_layer)
        
        if local_decoder is not None:
            self.layer_dict['local_decoder'] = local_decoder
            self.layer_dict['dropout'] = nn.Dropout(p=0.2)

        
        if reduction == 'gap':
            #  and uneven_t == False:
            self.layer_dict['gap'] = nn.AvgPool1d(max_len)
        self.layer_dict['classifier'] = classifier if classifier\
            is not None else nn.Linear(d_model, num_output_classes)
        

        self.reduction = reduction
        self.local_decoder = True if local_decoder is not None else False
        self.time_dimension = time_dimension
        self.uneven_t = uneven_t
        self.max_len = max_len
        self.d_model = d_model
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

    def forward(self, src):
        # if self.uneven_t: # seq lengths are different and thus are specified in dataset
        # print(type(src))
        if type(src) == list: # seq lengths are different and thus are specified in dataset
            lens = src[1]
            src = src[0]
            src = src.permute(0,2,1)
            pad_mask = self.pad_mask(src,lens)
            # print(pad_mask.shape)

        else:
            src = src.permute(0,2,1)
            pad_mask = torch.zeros((src.shape[0],src.shape[1]), device=torch.device('cuda'))
            # print(pad_mask.shape)

        if self.time_dimension:
            # x = src[:,:,0:-1] 
            t = src[:,:,-1] #last dimensions represent t
        # else:
        x = src
        
        x = self.layer_dict['embedding'](x) #input embedding 
        # x = x if not self.time_dimension else (x,t) 
        x = self.layer_dict['pos'](x) #positional encoding
        x = self.layer_dict['encoder'](x, src_key_padding_mask=pad_mask)
        
        try:
            #try to take all of the encoder representations as they are
            #this is in case the local decoder has dimensions to allow for it
            out = x.squeeze()
            if self.local_decoder:
                out = F.relu(self.layer_dict['local_decoder'](x))
                out = self.layer_dict['dropout'](out)

            out = out.view(x.shape[0],x.shape[1]*x.shape[2])
            out = self.layer_dict['classifier'](out)
        
        except Exception as e:

            if self.reduction is None:
                print("if no reduction is specified, then a custom classifier needs to be \
                    given to fit dimensions")
            
            elif self.reduction == 'last':
                    
                if self.uneven_t:
                    i =  lens.view(-1,1,1).long() - 1
                    i = i.repeat(1,1,self.d_model)

                    torch.set_printoptions(edgeitems=200)

                    out = x.gather(1,i) #last representation vector
                else:
                    out = x[:,-1,:] # if all sequences have the same length, take last element
            
            elif self.reduction == 'gap':

                # if self.uneven_t:
                #     out = torch.zeros(x.shape[0],x.shape[2], device=torch.device('cuda'))
                #     for i in torch.arange(0,x.shape[0]):
                #         out[i,:] = torch.mean(x[i,:lens[i],:],0)
                # else:
                out = x.permute(0,2,1)
                out = self.layer_dict['gap'](out)
                out = out.permute(0,2,1)


            if self.local_decoder:
                out = F.relu(self.layer_dict['local_decoder'](out))
                out = self.layer_dict['dropout'](out)

            out = out.squeeze()
            out = self.layer_dict['classifier'](out)

        return(out)

    def pad_mask(self, q, lens_q):
        # print(q.shape)
        # print(lens_q)
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


