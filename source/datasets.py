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
    #this function needs a data_loader that ensures that the appropiate indexes are asked for. that is
    #given a dataset of lenght L with that will be split in N chunks of size S, thsi data set will load into memmory
    #S objects at a time and the data_loade should ask for all indexes within that chunk first and then move on to the indexes of a different chunk.
    #otherwise it will still load but it will be extremely inefficient, as we will be loading chunks into memmory continually without reading them all.
    def __init__(self,lc_length, dataset_file, data_cache_size=100000, transform=None):

        self.lc_length = lc_length
        self.device = torch.device('cuda')
        self.data_cache = {}
        self.transform = transform
        self.data_cache_size = data_cache_size
        self.dataset_file = dataset_file
        self.dataset_length = 0
        self.n_chunks = 0
        self.low_idx = 0
        self.high_idx = 0
        self.chunks=[]
        try:
            with h5py.File(self.dataset_file,'r') as f:
                X = f["X"]
                Y = f["Y"]
                ids = f["ids"]
                self.dataset_length = len(ids)
                self.n_chunks = np.ceil(self.dataset_length/self.data_cache_size)

                self.chunks = list(np.arange(self.n_chunks))
                first_chunk=np.random.randint(len(self.chunks))
                self.current_chunk = first_chunk
                #pop this chunk from the list of available ones
                self.chunk.pop(first_chunk)

                self.low_idx = first_chunk*data_cache_size
                self.high_idx = (first_chunk+1)*data_cache_size if first_chunk != self.n_chunks else self.dataset_length
                self.chunks[str(self.current_chunk)]=True

                X = torch.tensor(X[self.low_idx:self.high_idx,:,0:self.lc_length], device = self.device, dtype=torch.float)
                Y = torch.tensor(Y[self.low_idx:self.high_idx], device = self.device, dtype=torch.long)
                ids = torch.tensor(self.ids[self.low_idx:self.high_idx], device = self.device, dtype=torch.int)
                batch_sample = X, Y, ids
                self.update_cache(batch_sample)
        except Exception as e:
            print(e)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if idx <= self.high_idx and idx >=self.low_idx: #if index asked for is in cache, return it
            idx = int(idx-self.low_idx)
            sample = self.data_cache['X'][idx], self.data_cache['Y'][idx], self.data_cache['ids'][idx]
            return sample
        else: # if index need is not in cache, need to load new chunk into memmory and call function again
            with h5py.File(self.dataset_file,'r') as f:
                new_chunk = np.random.randint(len(self.chunks))
                self.current_chunk = self.chunk.pop(new_chunk)
                self.low_idx = self.current_chunk*self.data_cache_size
                self.high_idx = (self.current_chunk+1)*self.data_cache_size if self.current_chunk != self.n_chunks else self.dataset_length
                X = f["X"][self.low_idx:self.high_idx,:,0:self.lc_length]
                Y = f["Y"][self.low_idx:self.high_idx]
                ids = f["ids"][self.low_idx:self.high_idx]
                X = torch.tensor(X, device = self.device, dtype=torch.float)
                Y = torch.tensor(Y, device = self.device, dtype=torch.long)
                ids = torch.tensor(ids, device = self.device, dtype=torch.int)
                batch_sample = X, Y, ids
                self.update_cache(batch_sample)
                self.__getitem__(idx)


    def update_cache(self,sample):
        self.data_cache['X'] = sample[0]
        self.data_cache['Y'] = sample[1]
        self.data_cache['ids'] = sample[2]


