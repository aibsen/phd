import torch
import numpy as np
import random

class RandomCrop(object):
  
    def __init__(self, output_size, lc_length):
        
        self.output_size = output_size
        self.lc_length = lc_length

    def __call__(self, sample):
        X,Y,obid=sample
        left = np.random.randint(0, self.lc_length - self.output_size)
        X=X[:,left:left+self.output_size]
        return X,Y,obid


class ZeroPad(object):

    def __init__(self, output_size, lc_length):    
        self.output_size = output_size
        self.lc_length = lc_length
        zeros = output_size-lc_length
        self.padding = torch.nn.ConstantPad1d((0,zeros), 0)

    def __call__(self, sample):
        X,Y,obid=sample
        if self.output_size > self.lc_length:
            X=self.padding(X)
        elif self.output_size<self.lc_length:
            X=X[:,0:self.output_size] #crop if no pad is needed
        return X,Y,obid


class RightCrop(object):
  
    def __init__(self, output_size, lc_length):
        
        self.output_size = output_size
        self.lc_length = lc_length

    def __call__(self, sample):
        if self.output_size <= self.lc_length:
            X,Y,obid=sample
            X=X[:,0:self.output_size]
            return X,Y,obid 
        else:
            print("crop size must be smaller than the length of lc")