from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
# import os
import numpy as np
import pandas as pd
import time
import sys
# from experiment import Experiment


class Seq2SeqExperiment():
    def __init__(self, parent):
        self.parent = parent

    def __getattr__(self, attr):
        return getattr(self.parent, attr)

    def run_train_iter(self, x, y,i):
        self.train()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        out = self.model.forward(x)  # forward the data in the model
        batch_size = x[0].shape[0]
        out = out.contiguous().view(batch_size,-1)
        tgt = x[0].contiguous().view(batch_size,-1)
        loss = self.criterion(out,tgt)
        loss.backward()  # backpropagate
        self.optimizer.step()
        return loss

    def run_evaluation_iter(self, x, y, i):
        self.eval()  # sets the system to validation mode
        out = self.model.forward(x)  # forward the data in the model
        batch_size = x[0].shape[0]
        out = out.contiguous().view(batch_size,-1)
        tgt = x[0].contiguous().view(batch_size,-1)
        loss = self.criterion(out,tgt)

        if i==0:
            original = tgt[0].view(-1,128).cpu().detach()
            reconstructed = out[0].view(-1,128).cpu().detach()
            l = x[1][0].cpu()
            plt.figure()
            plt.scatter(np.arange(l),original[0,-l:],label='original')
            plt.scatter(np.arange(l),reconstructed[0,-l:], label='reconstructed')
            plt.gca().invert_yaxis()
            plt.legend()
            plt.show()

        return loss

    def save_model(self, model_save_name):
        self.parent.save_model(model_save_name)

    def load_model(self, model_save_dir, model_save_name):
        self.parent.save_model(model_save_dir, model_save_name)

    def save_statistics(self, stats, fn):
        stats_keys = ['epoch', 'loss']
        self.parent.save_statistics(stats,fn,stats_keys)

    def run_test_phase(self, data=None, model_name="final_model.pth.tar",
        data_name="test",load_model=True):
        
        start_time = time.time()
        data = data if data else self.test_data
        
        if load_model:
            self.load_model(model_save_name=model_name)
        running_loss = 0.0
        
        with tqdm.tqdm(total=len(data)) as pbar:
            for i,(x, y, ids) in enumerate(data):
                loss = self.run_evaluation_iter(x=x,y=y)
                running_loss += loss.item()
                pbar.update(1)
                elapsed_time = time.time() - start_time 
                pbar.set_description("Test loss: {:.3f}        , ET {:.2f}s".format(running_loss/(i+1),elapsed_time))


        loss = running_loss/len(data)
        test_stats = torch.full((1,2),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch
        test_stats[0,0] = 1 #epoch
        test_stats[0,1] = loss #loss

        self.save_statistics(test_stats, data_name+"_summary.csv")

    def run_final_train_phase(self, data_loaders=None,n_epochs=None, 
        model_name="final_model.pth.tar", 
        data_name = 'final_training'):

        start_time = time.time()
        n_epochs = n_epochs if n_epochs else self.best_epoch
        data_loaders = data_loaders if data_loaders else [self.train_data,self.val_data]

        self.model.reset_parameters()

        train_stats = torch.full((n_epochs+1,2),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch

        n_batches = sum([len(data) for data in data_loaders])

        with tqdm.tqdm(total=n_epochs) as pbar_train:

            for i, epoch_idx in enumerate(range(n_epochs)):
                
                epoch_start_time = time.time()
                running_loss = 0.0
                cm_idx = 0    

                for i, data in enumerate(data_loaders):
                    for idx, (x, y, ids) in enumerate(data):
                        loss = self.run_train_iter(x=x, y=y, i=idx)
                        running_loss += loss.item()
                        batch_size = len(x)

                train_loss = running_loss/n_batches

                train_stats[epoch_idx,0] = epoch_idx+1 #epoch
                train_stats[epoch_idx,1] = train_loss #loss

                pbar_train.update(1)
                elapsed_time = time.time() - start_time  
                pbar_train.set_description("Train loss: {:.3f}       , ET {:.2f}s".format(train_loss,elapsed_time))
                
        self.save_statistics(train_stats, '{}_summary.csv'.format(data_name))
        self.state = {
            'epoch': n_epochs,
            'model': self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
                    }
        self.save_model(model_name) 
        
    def run_train_phase(self, model_name='best_validation_model.pth.tar', load_model=False):
        if load_model:
            self.load_model(model_save_name=model_name)

        start_time = time.time()

        strike = 0
        step_count = 0
        
        val_stats = torch.full((int(self.num_epochs/self.validation_step),2),-1, dtype=torch.float, device=self.device) # holds epoch,loss
        train_stats = torch.full((int(self.num_epochs),2),-1, dtype=torch.float, device=self.device) # holds epoch, loss
        
        train_n_batches = len(self.train_data)
        val_n_batches = len(self.val_data)
        val_loss = -1
        val_idx = 0

        with tqdm.tqdm(total=self.num_epochs) as pbar_train:

            for i, epoch_idx in enumerate(range(self.num_epochs)):
                step_count+=1
                running_train_loss = 0.0

                for idx, (x, y,ids) in enumerate(self.train_data):
                    # print(y)
                    loss = self.run_train_iter(x=x, y=y,i=idx)
                    
                    running_train_loss += loss.item()

                train_loss = running_train_loss/train_n_batches
                train_stats[epoch_idx,0] = epoch_idx+1 #epoch
                train_stats[epoch_idx,1] = train_loss #loss
            
                if step_count == self.validation_step:

                    step_count = 0
                    running_val_loss = 0.0

                    for idx, (x, y, ids) in enumerate(self.val_data):
                        loss = self.run_evaluation_iter(x=x, y=y, i=idx)
                        running_val_loss += loss.item()

                    val_loss = running_val_loss/val_n_batches
                    val_stats[val_idx,0] = epoch_idx+1 #epoch
                    val_stats[val_idx,1] = val_loss #loss
                    val_idx+=1

                    #if the f1-score of the current epoch for the validation set is better than previous epochs
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.best_epoch = epoch_idx
                        self.state = {
                            'epoch': epoch_idx,
                            'model': self.model.state_dict(),
                            'optimizer':self.optimizer.state_dict(),
                            'loss':val_loss
                        }
                        strike = 0

                    else:
                        strike+=1

                pbar_train.update(1)
                elapsed_time = time.time() - start_time 
                pbar_train.set_description("Tr/Val loss: {:.3f}/{:.3f}, Strike: {}, ET {:.2f}s".format(train_loss,val_loss,strike,elapsed_time))
                
                if strike == self.patience:
                    pbar_train.set_description("Tr/Val loss: {:.3f}/{:.3f}, Best epoch: {}, ET {:.2f}s".format(train_loss,val_loss,self.best_epoch+1,elapsed_time))
                    break

        #save statistics at the end only
        self.save_model("best_validation_model.pth.tar")
        self.save_statistics(train_stats, 'training_summary.csv')
        self.save_statistics(val_stats, 'validation_summary.csv')
    
    
    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        self.parent.run_experiment()