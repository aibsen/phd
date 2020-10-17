import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from torch.utils.data import Subset



def cached_dataset_indices_split(dataset, dataset_lengths, chunksize=100000):
    # print(dataset_lengths)
    if sum(dataset_lengths)>len(dataset):
        raise ValueError("Sum of input lengths is greater than the length of the dataset")

    n_chunks=np.ceil(len(dataset)/chunksize)
    rem = len(dataset)-(n_chunks-1)*chunksize #size of last chunk
    chunk_pool = [chunksize]*int(n_chunks-1)
    chunk_pool.append(rem)
    chunk_pool = list(zip(np.arange(len(chunk_pool)),chunk_pool))
    subsets_indices = []
    
    for length in dataset_lengths:
        # print(length)
        min_n_chunks = np.ceil(length/chunksize) #min n of chunks needed
        min_i_chunks = np.random.choice(np.arange(len(chunk_pool)),size=int(min_n_chunks),replace=False)
        min_chunks = [chunk_pool[i] for i in min_i_chunks] #list of tuples (index, size) randomly chosen chunks to be used
        rem_length = length
        indices = []

        for chunk in min_chunks:
            if rem_length >0:
                in_chunk = chunk[1]-rem_length #lo que me queda en el chunk - el length de data q todavía necesito
                chunk_idx = chunk_pool.index(chunk)
                shift = chunk[0]*chunksize
                if in_chunk <= 0 : #todavía me faltan índices y usé todo el chunk o no me quedan índices y usé todo el chunk
                    chunk_pool.pop(chunk_idx) #sacar chunk del pool
                    rem_length = in_chunk*(-1) #índices q todavía necesito para este split
                    idxs = np.arange(0,chunk[1])+shift #los índices para el split son desde 0 hasta lo q me quedaba en el chunk+el offset del chunk
                elif in_chunk > 0: #ya no me faltan índices para este split y me sobra chunk
                    idxs = np.arange(in_chunk,chunk[1])+shift
                    chunk_pool[chunk_idx] = (chunk[0], in_chunk) #marco lo que me queda en el chunk
                    rem_length = 0 
                indices = np.concatenate((indices,idxs),0)
        subsets_indices.append(indices)
    # print("SUBSET INDICES")
    # print(subsets_indices)
    return subsets_indices

def cached_dataset_random_split(dataset,dataset_lengths,chunksize=100000):
    subsets_indices=cached_dataset_indices_split(dataset,dataset_lengths,chunksize)
    return [Subset(dataset, idx) for idx in subsets_indices]


def cached_crossvalidator_split(dataset,dataset_lengths,chunksize=100000):
    subsets_indices=cached_dataset_indices_split(dataset,dataset_lengths)
    l=len(dataset)
    for val_index in subsets_indices:
        indices = np.arange(l)
        # print(val_index)
        validation_index=val_index.astype(int)
        train_index = np.delete(indices,val_index)
        # print(train_index)
        yield train_index, validation_index
