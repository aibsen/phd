import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import matplotlib.pyplot as plt
import itertools as it
import h5py
import pandas as pd

class Interpolated_LCs(Dataset):
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


class Interpolated_LCs_C(Dataset):
    def __init__(self, lc_length,classes_to_consider,dataset_h5):

        self.lc_length = lc_length
        self.dataset_h5 = dataset_h5
        self.device = torch.device('cuda')
        self.X = None
        self.Y = None
        self.ids = None


        try:
            with h5py.File(self.dataset_h5) as f:
                X = f["X"][:,:,0:self.lc_length]
                Y = f["Y"]
                ids = f["ids"]
                self.X = torch.tensor(X, device = self.device, dtype=torch.float)
                self.ids = torch.tensor(ids, device = self.device, dtype=torch.int)
                self.Y = self.less_classes(classes_to_consider,torch.tensor(Y, device = self.device, dtype=torch.long))
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

    def less_classes(self, classes_to_consider,Y):
        # print(Y)
        #retag stuff so we only consider certain classes and everything else is tagged as 'other'
        #should probs refactor this to make it less loopy
        l = len(classes_to_consider)
        for i,y in enumerate(Y):
            if y not in classes_to_consider:
                Y[i]=l
            else:
                for j,c in enumerate(classes_to_consider):
                    if y == c:
                        Y[i]=j
        # print(Y)
        return Y

    def get_tags(self):
        return self.Y



class RandomCrop(object):
  
    def __init__(self, output_size, lc_length):
        
        self.output_size = output_size
        self.lc_length = lc_length

    def __call__(self, sample):
        X,Y,obid=sample
        left = np.random.randint(0, self.lc_length - self.output_size)
        X=X[:,left:left+self.output_size]
        return X,Y,obid

# import h5py
# import helpers
# from pathlib import Path
#
# class HDF5Dataset(data.Dataset):
#     """Represents an abstract HDF5 dataset.
#
#     Input params:
#         file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
#         recursive: If True, searches for h5 files in subdirectories.
#         load_data: If True, loads all the data immediately into RAM. Use this if
#             the dataset is fits into memory. Otherwise, leave this at false and
#             the data will load lazily.
#         data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
#         transform: PyTorch transform to apply to every data instance (default=None).
#     """
#     def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
#         super().__init__()
#         self.data_info = []
#         self.data_cache = {}
#         self.data_cache_size = data_cache_size
#         self.transform = transform
#
#         # Search for all h5 files
#         p = Path(file_path)
#         assert(p.is_dir())
#         if recursive:
#             files = sorted(p.glob('**/*.h5'))
#         else:
#             files = sorted(p.glob('*.h5'))
#         if len(files) < 1:
#             raise RuntimeError('No hdf5 datasets found')
#
#         for h5dataset_fp in files:
#             self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
#
#     def __getitem__(self, index):
#         # get data
#         x = self.get_data("data", index)
#         if self.transform:
#             x = self.transform(x)
#         else:
#             x = torch.from_numpy(x)
#
#         # get label
#         y = self.get_data("label", index)
#         y = torch.from_numpy(y)
#         return (x, y)
#
#     def __len__(self):
#         return len(self.get_data_infos('data'))
#
#     def _add_data_infos(self, file_path, load_data):
#         with h5py.File(file_path) as h5_file:
#             # Walk through all groups, extracting datasets
#             for gname, group in h5_file.items():
#                 for dname, ds in group.items():
#                     # if data is not loaded its cache index is -1
#                     idx = -1
#                     if load_data:
#                         # add data to the data cache
#                         idx = self._add_to_cache(ds.value, file_path)
#
#                     # type is derived from the name of the dataset; we expect the dataset
#                     # name to have a name such as 'data' or 'label' to identify its type
#                     # we also store the shape of the data in case we need it
#                     self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})
#
#     def _load_data(self, file_path):
#         """Load data to the cache given the file
#         path and update the cache index in the
#         data_info structure.
#         """
#         with h5py.File(file_path) as h5_file:
#             for gname, group in h5_file.items():
#                 for dname, ds in group.items():
#                     # add data to the data cache and retrieve
#                     # the cache index
#                     idx = self._add_to_cache(ds.value, file_path)
#
#                     # find the beginning index of the hdf5 file we are looking for
#                     file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)
#
#                     # the data info should have the same index since we loaded it in the same way
#                     self.data_info[file_idx + idx]['cache_idx'] = idx
#
#         # remove an element from data cache if size was exceeded
#         if len(self.data_cache) > self.data_cache_size:
#             # remove one item from the cache at random
#             removal_keys = list(self.data_cache)
#             removal_keys.remove(file_path)
#             self.data_cache.pop(removal_keys[0])
#             # remove invalid cache_idx
#             self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]
#
#     def _add_to_cache(self, data, file_path):
#         """Adds data to the cache and returns its index. There is one cache
#         list for every file_path, containing all datasets in that file.
#         """
#         if file_path not in self.data_cache:
#             self.data_cache[file_path] = [data]
#         else:
#             self.data_cache[file_path].append(data)
#         return len(self.data_cache[file_path]) - 1
#
#     def get_data_infos(self, type):
#         """Get data infos belonging to a certain type of data.
#         """
#         data_info_type = [di for di in self.data_info if di['type'] == type]
#         return data_info_type
#
#     def get_data(self, type, i):
#         """Call this function anytime you want to access a chunk of data from the
#             dataset. This will make sure that the data is loaded in case it is
#             not part of the data cache.
#         """
#         fp = self.get_data_infos(type)[i]['file_path']
#         if fp not in self.data_cache:
#             self._load_data(fp)
#
#         # get new cache_idx assigned by _load_data_info
#         cache_idx = self.get_data_infos(type)[i]['cache_idx']
#         return self.data_cache[fp][cache_idx]
