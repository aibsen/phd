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
        # print("DATASEt length in sampler"+str(len(data_source)))
        self.dataset_length = len(data_source)
        self.chunk_size = chunk_size 
        # print(data_source.indices)
        self.indices = data_source.indices
        self.current_chunk = np.floor(self.indices[0]/self.chunk_size)

    def __iter__(self):
        indices_in_chunk = 0
        it = torch.tensor([],dtype = torch.long)
        # print(self.indices)
        for i in self.indices:
            current_chunk = np.floor(i/self.chunk_size)
            # print(current_chunk) 
            if current_chunk == self.current_chunk:
                indices_in_chunk = indices_in_chunk+1
            else:
                # print(indices_in_chunk)    
                shift = self.current_chunk*self.chunk_size
                index_order = torch.randperm(int(indices_in_chunk))
                indices = index_order+shift
                # print(indices)
                it = torch.cat((it,indices))
                self.current_chunk = current_chunk
                indices_in_chunk = 1
        shift = self.current_chunk*self.chunk_size
        index_order = torch.randperm(int(indices_in_chunk))
        indices = index_order+shift
        # print(indices)
        it = torch.cat((it,indices))
        # print("it")
        # print(it.tolist())
        return iter(it.tolist())
        # return iter(self.indices)

    def __len__(self):
        return self.dataset_length