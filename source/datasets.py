import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import random

class LCs(Dataset):
    def __init__(self, lc_length, dataset_h5,n_channels=12,transforms=None, packed=False):

        self.lc_length = lc_length
        self.dataset_h5 = dataset_h5
        self.device = torch.device('cuda')
        self.X = None
        self.Y = None
        self.ids = None
        self.transforms = transforms
        self.length = None
        self.n_channels = n_channels
        self.targets = None
        self.lens = None
        self.packed = packed

    def __len__(self):
        # # if self.length is None:
        #     print("Data needs to be loaded to memmory first, use method load_data_into_memmory()")
        return self.length

    def __getitem__(self, idx):
    #     if self.X is None:
    #         self.load_data_into_memory()
        if self.packed:
            return (self.X[idx],self.lens[idx]),self.Y[idx], self.ids[idx]
        return self.X[idx],self.Y[idx], self.ids[idx]
        
    # def __getitem__(self, idx):
        # return self.X[idx],self.Y[idx], self.ids[idx]


    def load_data_into_memory(self):
        print("loading data into memmory")
        try:
            with h5py.File(self.dataset_h5,'r') as f:
                X = f["X"][:,:,0:self.lc_length]
                print(X.shape)
                Y = f["Y"]
                ids = f["ids"]
                self.targets = list(Y)
                # print(self.targets)
                self.X = torch.tensor(X, device = self.device, dtype=torch.float)
                self.ids = torch.tensor(ids, device = self.device, dtype=torch.long)
                self.Y = torch.tensor(Y, device = self.device, dtype=torch.long)
                self.length = len(self.Y)
                if 'lens' in f.keys():
                    lens=f['lens']
                    self.lens=torch.tensor(lens, device = self.device, dtype=torch.int64)
                    # print(self.lens.dtype)
                # print(self.X.shape)
                # print(self.Y.shape)
                # print(self.ids.shape)
                # print(len(set(self.targets)))
        except Exception as e:
            print(e)



    def apply_tranforms(self):
        if self.transforms:
            for transform in self.transforms:
                sample = self.X,self.Y, self.ids
                self.X, self.Y, self.ids, self.lens = transform(sample)


    def get_items(self,idxs):
        X = self.X[idxs]
        Y = self.Y[idxs]
        ids = self.ids[idxs]
        return X, Y, ids         

