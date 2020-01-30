import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import matplotlib.pyplot as plt
import itertools as it
import h5py
import pandas as pd

class Interpolated_LCs(Dataset):
    def __init__(self, lc_length, dataset_h5):

        self.lc_length = lc_length
        self.dataset_h5 = dataset_h5
        self.device = torch.device('cuda')
        self.X = None
        self.Y = None
        self.ids = None


        try:
            with h5py.File(self.dataset_h5) as f:
                X = f["X"][:,:,0:self.lc_length]
                print(X.shape)
                Y = f["Y"]
                print(Y.shape)
                ids = f["ids"]
                print(ids.shape)
                self.X = torch.tensor(X, device = self.device, dtype=torch.float)
                self.ids = torch.tensor(ids, device = self.device, dtype=torch.int)
                self.Y = torch.tensor(Y, device = self.device, dtype=torch.long)
        except Exception as e:
            print(e)
    
    def __len__(self):
        if self.ids is not None:
            return len(self.ids)
        else:
            return 0

    def __getitem__(self, idx):
        if self.X is not None:
            return self.X[idx],self.Y[idx]
        else:
            return None

