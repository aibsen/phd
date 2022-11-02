import numpy as np
import torch
from torch import nn, Tensor
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128, custom_position=False):
        super().__init__()
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['dropout'] = nn.Dropout(p=dropout)
        self.custom_position = custom_position
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.max_len = max_len
        self.d_model = d_model

        if not custom_position:
            position = torch.arange(max_len).unsqueeze(1)
            
            # pe = torch.zeros(max_len, 1, d_model)
            pe = torch.zeros(1,max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            # pe = pe.permute(1,0,2)
            self.register_buffer('pe', pe)
        else:
            self.register_buffer('div_term',div_term)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, embedding_dim]
        """
        if self.custom_position: 
            #if sequences are not evenly sampled, then the last f_dim represents custom positions
            positions = x[-1].unsqueeze(-1)
            x = x[0]
            l = positions*self.div_term
            pe = torch.zeros((x.shape[0],self.max_len, self.d_model),device=torch.device('cuda'))
            pe[:, :, 0::2] = torch.sin(positions * self.div_term)
            pe[:, :, 1::2] = torch.cos(positions * self.div_term)
            x = x + pe

        else:
            x = x + self.pe[:x.size(0)]

        return self.layer_dict['dropout'](x)
    
    def reset_parameters(self):
        pass

class FourierDecomposition(nn.Module):
    
    def __init__(self, k=128, harmonics=12,kt = 1.5):

        sin_weights = torch.Tensor((k,harmonics))
        cos_weights = torch.Tensor((k,harmonics))

        self.kt = kt
        self.sin_weights = nn.Parameter(sin_weights)
        self.cos_weights = nn.Parameter(cos_weights)
        self.harmonics = nn.range(harmonics+1)[1:]

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.sin_weights)
        nn.init.kaiming_normal_(self.cos_weights)

    def reset_parameters(self) -> None:
        self.init_weights()

    def forward(self, t):
        t_max = self.kt*torch.max(t)
        polar = (2*np.pi*self.harmonics*t)/t_max
        sin_part = self.sin_weights*nn.sine(polar)
        cos_part = self.cos_weights*nn.sine(polar)
        out = torch.add(sin_part,cos_part)
        return out


class TimeFiLMEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, n_bands=2):
        super().__init__()

        self.layer_dict = nn.ModuleDict()
        self.layer_dict['input_projection'] = nn.Linear(1+n_bands,d_model,bias=False)
        self.layer_dict['scale'] = FourierDecomposition(k=d_model)
        self.layer_dict['bias'] = FourierDecomposition(k=d_model)
        self.layer_dict['output_modulation'] = nn.Linear(d_model, d_model)
        self.init_weights()

    def init_weights(self) -> None:
        for i, item in enumerate(self.layer_dict.children()):
            try:
                item.reset_parameters()
            except Exception as e:
                print(e)

    def reset_parameters(self) -> None:
        self.init_weights()

    def forward(self, x: Tensor, lens: Tensor) -> Tensor:

        out = self.layer_dict['input_projection'](x)
        out = torch.mul(torch.tanh(self.layer_dict['scale'](lens)),out)
        out = torch.add(out,self.layer_dict['bias'](lens))
        out = nn.functional.relu(self.layer_dict['output_modulation'](out))
        return out