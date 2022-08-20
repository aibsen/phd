import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import random

class LCs(Dataset):
    def __init__(self, lc_length, dataset_h5,n_channels=12,transforms=None, n_classes=14,packed=False):

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
        self.n_classes=n_classes
        self.lens = None
        self.packed = packed

        # try:
        #     with h5py.File(self.dataset_h5,'r') as f:
        #         Y = f["Y"]
        #         self.length = len(Y)
        #         self.targets = list(Y)
              
        # except Exception as e:
        #     print(e)

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

    def filter_classes(self,classes_to_keep=None):
        if self.X is None:
            self.load_data_into_memory()

        if classes_to_keep is None:
            classes_to_keep = self.n_classes

        if type(classes_to_keep) == int:
            classes_to_keep = range(classes_to_keep)
        try:
            self.n_classes = len(classes_to_keep)
            classes_to_keep = torch.tensor(classes_to_keep,device=self.device)
            idxs = torch.where((torch.isin(self.Y,classes_to_keep)))[0] 
            self.Y = self.Y[idxs]
            self.X = self.X[idxs]
            self.ids = self.ids[idxs]
            self.lens = None if self.lens is None else self.lens[idxs]
            self.length = len(self.Y)
            self.targets = list(self.Y.cpu())
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)

    def load_data_into_memory(self):
        print("loading data into memmory")
        try:
            with h5py.File(self.dataset_h5,'r') as f:
                X = f["X"][:,0:self.n_channels]
                Y = f["Y"]
                ids = f["ids"]
                self.targets = list(Y)
                self.X = torch.tensor(X, device = self.device, dtype=torch.float)
                self.ids = torch.tensor(ids, device = self.device, dtype=torch.long)
                self.Y = torch.tensor(Y, device = self.device, dtype=torch.long)
                self.length = len(self.Y)
        except Exception as e:
            print(e)

    def get_samples_per_class(self):
        return [self.targets.count(i) for i in range(self.n_classes)]

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

