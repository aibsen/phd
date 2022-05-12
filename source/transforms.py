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



class RandomCropsZeroPad(object):
  
    def __init__(self, output_sizes, lc_length):
        
        self.output_sizes = output_sizes
        self.lc_length = lc_length

    def __call__(self, sample):
        X,Y,obid=sample
        size_idx = torch.randint(0,len(self.output_sizes),(1,)) #fix this so it works with batches
        size = self.output_sizes[size_idx]
        left = 0 if self.lc_length==size else torch.randint(0,(self.lc_length - size),(1,))
        X=X[:,left:left+size]
        #now zero pad if necessary
        if size < self.lc_length:
            zeros = self.lc_length-size
            padding = torch.nn.ConstantPad1d((0,zeros),0)
            X=padding(X)
        return X,Y,obid


class CastClass(object):

    def __init__(initial_class,final_class):
        self.initial_class = initial_class
        self.final_class = final_class

    def __call__(self,sample):
        X,Y,obid=sample
        if Y == self.initial_class:
            Y=self.final_class
        return X,Y,obid