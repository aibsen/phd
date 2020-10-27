import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import random

class LCs(Dataset):
    def __init__(self, lc_length, dataset_h5,n_channels=2,transform=None):

        self.lc_length = lc_length
        self.dataset_h5 = dataset_h5
        self.device = torch.device('cuda')
        self.X = None
        self.Y = None
        self.ids = None
        self.transform = transform
        self.length = None
        self.n_channels = n_channels

        try:
            with h5py.File(self.dataset_h5,'r') as f:
                X = f["X"][:,0:self.n_channels,0:self.lc_length]
                Y = f["Y"]
                ids = f["ids"]
                print(X[0].shape)
                print(len(X))
                self.length = len(X)
              
        except Exception as e:
            print(e)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.X is None:
            self.load_data_into_memory()
        sample = self.X[idx],self.Y[idx], self.ids[idx]
        if self.transform:
            return self.transform(sample)
        else:
            return sample

    def load_data_into_memory(self):
        try:
            with h5py.File(self.dataset_h5,'r') as f:
                X = f["X"][:,0:self.n_channels,0:self.lc_length]
                Y = f["Y"]
                ids = f["ids"]
                self.X = torch.tensor(X, device = self.device, dtype=torch.float)
                self.ids = torch.tensor(ids, device = self.device, dtype=torch.int)
                self.Y = torch.tensor(Y, device = self.device, dtype=torch.long)
        except Exception as e:
            print(e)

    def get_samples_per_class(self,n_classes):
        if self.Y is None:
            self.load_data_into_memory()
        self.n_classes = n_classes
        counts = torch.zeros(n_classes)
        for i in np.arange(n_classes):
            idx = torch.where(self.Y == i)[0]
            counts[i] = len(self.Y[idx])
        return counts

    def get_all_labels(self):
        if self.Y is None:
            self.load_data_into_memory()
        return self.Y

    def get_items(self,idxs):
        X = self.X[idxs]
        Y = self.Y[idxs]
        ids = self.ids[idxs]
        return X, Y, ids         


class InefficientCachedLCs(Dataset):

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


class CachedLCs(Dataset):

    def __init__(self,lc_length, dataset_file, chunk_size=100000, dataset_length=None, indices=None, transform=None):

        self.lc_length = lc_length
        self.device = torch.device('cuda')

        self.chunk_size = chunk_size
        self.dataset_file = dataset_file

        self.X = None
        self.Y = None
        self.ids = None

        self.transform = transform
        self.dataset_length = dataset_length
        self.true_dataset_length = None
        self.indices = indices

        self.low_idx = 0
        self.high_idx = -1
        self.loading_data=0

        try:
            with h5py.File(self.dataset_file,'r') as f:
                X = f["X"]
                Y = f["Y"]
                ids = f["ids"]
                self.true_dataset_length = len(ids)
                if self.dataset_length is None:
                    self.dataset_length = len(ids)
                if self.indices is None:
                    self.dataset_indices = np.arange(0,self.dataset_length)

        except Exception as e:
            print(e)

        # print(self.indices)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # print(idx)
        if idx <= self.high_idx and idx >=self.low_idx: #if index asked for is in cache, return it
            idx = int(idx-self.low_idx)
            sample = self.X[idx], self.Y[idx], self.ids[idx]
        else: #if index asked for is not in cache, load it
            print("loading data")
            self.loading_data=self.loading_data+1
            print(self.loading_data)
            with h5py.File(self.dataset_file,'r') as f:
                # current_chunk = np.floor(idx/self.chunk_size)
                current_chunk = torch.floor(torch.tensor(idx/self.chunk_size,device=self.device))
                self.low_idx = int(current_chunk*self.chunk_size)
                high_idx = int((current_chunk+1)*self.chunk_size)
                self.high_idx =int((current_chunk+1)*self.chunk_size) if high_idx<self.true_dataset_length else int(self.true_dataset_length-1)
                # stats = torch.cuda.memory_allocated()
                # print("low : "+str(self.low_idx)+" < "+str(idx)+" high: "+str(self.high_idx))
                # print("STATS before LOADING DATA ··················")
                # print(stats)
                del self.X
                del self.Y
                del self.ids
                torch.cuda.empty_cache()
        
                self.X = torch.tensor(f["X"][self.low_idx:self.high_idx,:,0:self.lc_length], device = self.device, dtype=torch.float)
                self.Y = torch.tensor(f["Y"][self.low_idx:self.high_idx], device = self.device, dtype=torch.long)
                self.ids = torch.tensor(f["ids"][self.low_idx:self.high_idx], device = self.device, dtype=torch.int)
                # print(self.X.size())
                # stats = torch.cuda.memory_allocated()
                # print("STATS after LOADING DATA ··················")
                # print(stats)
                idx = int(idx-self.low_idx)
                # print(idx)
                # print(self.X[idx].size())
                # print(self.Y[idx])
                sample = self.X[idx], self.Y[idx], self.ids[idx]

        if self.transform:
            # print("hay transform")
            # print(self.transform)
            # print(sample)
            return self.transform(sample)
        else:
            return sample
