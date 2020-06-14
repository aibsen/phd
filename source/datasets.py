import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import random

class LCs(Dataset):
    def __init__(self, lc_length, dataset_h5,transform=None):

        self.lc_length = lc_length
        self.dataset_h5 = dataset_h5
        self.device = torch.device('cuda')
        self.X = None
        self.Y = None
        self.ids = None
        self.transform = transform

        try:
            with h5py.File(self.dataset_h5,'r') as f:
                X = f["X"][:,:,0:self.lc_length]
                Y = f["Y"]
                ids = f["ids"]
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
            sample = self.X[idx],self.Y[idx], self.ids[idx]
            if self.transform:
                return self.transform(sample)
            else:
                return sample
        else:
            return None

class CachedLCs(Dataset):

    def __init__(self,lc_length, dataset_file, data_cache_size=100000, transform=None):

        self.lc_length = lc_length
        self.device = torch.device('cuda')
        self.data_cache = {}
        self.transform = transform
        self.data_cache_size = data_cache_size
        self.dataset_file = dataset_file
        self.dataset_length = 0
        
        try:
            with h5py.File(self.dataset_file,'r') as f:
                X = f["X"]
                Y = f["Y"]
                ids = f["ids"]
                self.dataset_length = len(ids)

                for i in np.arange(self.data_cache_size):
                    X = torch.tensor(X[i,:,0:self.lc_length], device = self.device, dtype=torch.float)
                    Y = torch.tensor(Y[i], device = self.device, dtype=torch.long)
                    ids = torch.tensor(ids[i], device = self.device, dtype=torch.int)
                    sample = X, Y, ids
                    self.add_to_cache(sample, i)
        except Exception as e:
            print(e)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        idx = str(idx)
        if idx in self.data_cache:
            sample = self.data_cache[idx]
        else:
            with h5py.File(self.dataset_file,'r') as h5_file:
                idx = int(idx)
                X = h5_file["X"][idx,:,0:self.lc_length]
                Y = h5_file["Y"][idx]
                ids = h5_file["ids"][idx]
                X = torch.tensor(X, device = self.device, dtype=torch.float)
                Y = torch.tensor(Y, device = self.device, dtype=torch.long)
                ids = torch.tensor(ids, device = self.device, dtype=torch.int)
                sample = X, Y, ids
                self.add_to_cache(sample, idx)

        if self.transform:
            return self.transform(sample)
        else:
            return sample

    def add_to_cache(self,sample, idx):
        #if cache is full, delete random element
        if len(self.data_cache) + 1 > self.data_cache_size:
            del self.data_cache[random.choice(list(self.data_cache))]
        self.data_cache[str(idx)] = sample


