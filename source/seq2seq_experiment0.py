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
import plot_utils
import preprocess_data_utils
# from experiment import Experiment


class Seq2SeqExperiment():
    def __init__(self, parent, consider_label=True, unsupervised=False):
        self.parent = parent
        self.consider_label = consider_label
        self.unsupervised = unsupervised
        if self.consider_label:
            self.classification_loss = nn.CrossEntropyLoss().to(self.device) 

    def __getattr__(self, attr):
        return getattr(self.parent, attr)


    def criterion(self, output,tgt_expected, y_pred=None,y=None):
        reconstruction_loss = self.parent.criterion(output,tgt_expected)
        if self.consider_label:
            # print("it's the kid!")
            return reconstruction_loss + 1E-4*self.classification_loss(y_pred,y)
        else: 
            return reconstruction_loss

    def run_train_iter(self, x, y=None):
        self.train()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        
        src = x[0]
        tgt = x[0].new_zeros(x[0].shape)
        tgt[:,:,1:] = x[0][:,:,:-1]
        tgt[:,:,0] = -1
        tgt_expected = src

        out, y_pred = self.model.forward(src,tgt)  # forward the data in the model

        batch_size = src.shape[0]
        out = out.contiguous().view(batch_size,-1)
        tgt_expected = tgt_expected.contiguous().view(batch_size,-1)

        loss = self.criterion(out,tgt_expected,y_pred,y)
        loss.backward()  # backpropagate
        self.optimizer.step()

        return loss

    def run_evaluation_iter(self, x, y=None):
        self.eval()
        with torch.no_grad():
            src = x[0]
            tgt = x[0].new_zeros(x[0].shape)
            tgt[:,:,1:] = x[0][:,:,:-1]
            tgt[:,:,0] = -1
            tgt_expected = src

            out, y_pred = self.model.forward(src,tgt)  # forward the data in the model

            sample_reconstructed = out[0].cpu().detach()
            batch_size = src.shape[0]
            out_reshape = out.contiguous().view(batch_size,-1)
            tgt_expected = tgt_expected.contiguous().view(batch_size,-1)

            loss = self.criterion(out_reshape,tgt_expected,y_pred,y)
            return loss, sample_reconstructed, out


    def predict(self, x, max_length=128):
        self.eval()
        
        input_sequence = x[0]
        # print(x[0].shape)
        y_input = torch.tensor([[[0]]], dtype=torch.float, device=self.device).repeat(input_sequence.shape[0],input_sequence.shape[1],input_sequence.shape[2])
        y_input[:,:,0]=-1
        # y_input = torch.tensor([[[SOS_token]]], dtype=torch.float, device=self.device).repeat(input_sequence.shape[0],input_sequence.shape[1],1)
        # print(y_input.shape)
        num_tokens = len(input_sequence[0])

        with torch.no_grad():
            for _ in range(max_length):
                # Get source mask
                # tgt_mask = self.model.generate_square_subsequent_mask(y_input.size(1))
                
                y_input = self.model(input_sequence, y_input)#, tgt_mask)
                # print(pred.shape)
            # print(pred[0])

            
            # next_item = pred.topk(1, dim=2)[0].detach()
            # print(next_item[0])
            # print(next_item.shape)
            # [1].view(-1)[-1].item() # num with highest probability
            # next_item = torch.tensor([[next_item]], device=self.device)
            # print(y_input.shape)
        #     # Concatenate previous input with predicted best word
            # y_input = torch.cat((y_input, next_item), dim=2)
            # print(y_input.shape)
            # print(next_item)
            # break


        #     # Stop if model predicts end of sentence
        #     if next_item.view(-1).item() == EOS_token:
        #         break
        batch_size = input_sequence.shape[0]
        y_input = y_input.contiguous().view(batch_size,-1)
        tgt_expected = input_sequence.contiguous().view(batch_size,-1)
        loss = self.criterion(y_input,tgt_expected)
        print(loss)
        # print(y_input[0])
        # print(y_input.shape)
        # return y_input.view(-1).tolist()


    def save_model(self, model_save_name):
        self.parent.save_model(model_save_name)

    def load_model(self, model_save_name):
        self.parent.load_model(model_save_name)

    def save_statistics(self, stats, fn):
        stats_keys = ['epoch', 'loss']
        self.parent.save_statistics(stats,fn,stats_keys)

    def run_test_phase(self, data=None, model_name="final_model.pth.tar",
        data_name="test",load_model=True):
        
        start_time = time.time()
        data = data if data else self.test_data
        
        if load_model:
            print("loading model")
            self.load_model(model_save_name=model_name)
        running_loss = 0.0
        
        test_dataset_length = len(data.dataset)
        n_channels = data.dataset[0][0][0].shape[0]
        max_lc_length = data.dataset[0][0][0].shape[1]

        reconstructed_x = torch.zeros(test_dataset_length,n_channels,max_lc_length)
        lens = torch.zeros(test_dataset_length)
        reconstructed_y = torch.zeros(test_dataset_length)
        reconstructed_ids = torch.zeros(test_dataset_length)

        with tqdm.tqdm(total=len(data)) as pbar:
            for i,(x, y, ids) in enumerate(data):
                loss, random_sample, out = self.run_evaluation_iter(x=x,y=y)
                running_loss += loss.item()

                reconstructed_x[i*self.batch_size:i*self.batch_size+out.shape[0]] = out.detach().cpu() 
                reconstructed_y[i*self.batch_size:i*self.batch_size+out.shape[0]] = y.cpu()
                reconstructed_ids[i*self.batch_size:i*self.batch_size+out.shape[0]] =ids.cpu()
                lens[i*self.batch_size:i*self.batch_size+out.shape[0]] = x[1].cpu()

                
                pbar.update(1)
                elapsed_time = time.time() - start_time 
                pbar.set_description("Test loss: {:.3f}        , ET {:.2f}s".format(running_loss/(i+1),elapsed_time))


        loss = running_loss/len(data)
        test_stats = torch.full((1,2),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch
        test_stats[0,0] = 1 #epoch
        test_stats[0,1] = loss #loss

        results_dataset = {
            'X': reconstructed_x,
            'Y': reconstructed_y,
            'ids': reconstructed_ids,
            'lens': lens
        }

        preprocess_data_utils.save_vectors(results_dataset,self.parent.experiment_logs+'/'+'reconstructed_{}.h5'.format(data_name))
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
                        loss = self.run_train_iter(x=x, y=y)
                        running_loss += loss.item()
                        batch_size = len(x)

                train_loss = running_loss/n_batches

                train_stats[epoch_idx,0] = epoch_idx+1 #epoch
                train_stats[epoch_idx,1] = train_loss #loss

                pbar_train.update(1)
                elapsed_time = time.time() - start_time  
                pbar_train.set_description("Train loss: {:.3f}       , ET {:.2f}s".format(train_loss,elapsed_time))
                
        self.save_statistics(train_stats, '{}_summary.csv'.format(data_name))
        self.parent.state = {
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


        random_samples =[]
        with tqdm.tqdm(total=self.num_epochs) as pbar_train:

            for i, epoch_idx in enumerate(range(self.num_epochs)):
                step_count+=1
                running_train_loss = 0.0

                for idx, (x, y,ids) in enumerate(self.train_data):
                    # print(y)
                    loss = self.run_train_iter(x=x, y=y)
                    
                    running_train_loss += loss.item()

                train_loss = running_train_loss/train_n_batches
                train_stats[epoch_idx,0] = epoch_idx+1 #epoch
                train_stats[epoch_idx,1] = train_loss #loss
            
                if step_count == self.validation_step:

                    step_count = 0
                    running_val_loss = 0.0

                    for idx, (x, y, ids) in enumerate(self.val_data):


                        # loss = self.run_evaluation_iter(x=x, y=y, i=idx)
                        loss, random_sample, out = self.run_evaluation_iter(x=x, y=y)
                        running_val_loss += loss.item()

                        # self.predict(x)
                        if idx == len(self.val_data)-1:
                            random_example = [x[0][0].cpu().detach(), random_sample,x[1][0].cpu(),ids[0].cpu(),y[0].cpu(), epoch_idx] 
                            #original, reconstructed, length, id, class, epoch 
                            random_samples.append(random_example)
                            

                    val_loss = running_val_loss/val_n_batches
                    val_stats[val_idx,0] = epoch_idx+1 #epoch
                    val_stats[val_idx,1] = val_loss #loss
                    val_idx+=1

                    #if the f1-score of the current epoch for the validation set is better than previous epochs
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.best_epoch = epoch_idx
                        self.parent.state = {
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

        #save statistics at the end onlystats_df.to_csv(self.experiment_logs+'/'+fn ,sep=',',index=False)
        print("saving reconstruction results...")
        plot_utils.plot_reconstructions_per_epoch(random_samples, self.validation_step, self.best_epoch, self.experiment_logs+'/'+"reconstruction_per_epoch.png")
        self.save_model("best_validation_model.pth.tar")
        self.save_statistics(train_stats, 'training_summary.csv')
        self.save_statistics(val_stats, 'validation_summary.csv')
    
    
    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        self.parent.run_experiment()