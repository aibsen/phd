import torch
import numpy as np
from torch.utils.data import Sampler
import h5py
import random

class CachedRandomSampler(Sampler):
    """Samples elements randomly from a random chunk of data loaded in memory
    
    Arguments:
        data_source : dataset to sample from
        chunk_size : size of chunks that will be loaded into memory,
        in terms of number of objects
    """

    def __init__(self, data_source, chunk_size=100000):
        self.dataset_length = len(data_source)
        self.chunk_size = chunk_size 
        self.n_chunks = np.ceil(self.dataset_length/self.chunk_size)

    def __iter__(self):
        chunk_order = torch.randperm(int(self.n_chunks))
        it = torch.tensor([],dtype = torch.long)
        print(chunk_order)
        print(self.n_chunks)
        for i,chunk in enumerate(chunk_order):
            print(chunk)
            shift = chunk*self.chunk_size
            if chunk == self.n_chunks-1:
                limit = self.dataset_length-(self.n_chunks-1)*self.chunk_size
                index_order = torch.randperm(int(limit))
                print(limit)
            else:
                index_order = torch.randperm(int(self.chunk_size))
            indexes = index_order+shift
            # print(indexes)
            it = torch.cat((it,indexes),0)
        # print(it.tolist())
        return iter(it.tolist())

    def __len__(self):
        return self.dataset_length
