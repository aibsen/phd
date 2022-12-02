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


    def run_train_iter(self, x, y=None, i=0):
        self.train()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        
        src = x[0]
        tgt = x[0].new_zeros(x[0].shape)
        tgt[:,:,1:] = x[0][:,:,:-1]
        tgt[:,:,0] = -1
        tgt_expected = src

        out, y_pred, mu_z, logvar_z  = self.model.forward(src,tgt)  # forward the data in the model

        batch_size = src.shape[0]
        out_reshape = out.contiguous().view(batch_size,-1)
        tgt_expected = tgt_expected.contiguous().view(batch_size,-1)

        loss = self.criterion(out_reshape,tgt_expected, mu_z, logvar_z, y_pred, y, i)
        loss.backward()  # backpropagate
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        return loss

    def run_evaluation_iter(self, x, y=None,i=1):
        self.eval()
        with torch.no_grad():
            src = x[0]
            tgt = x[0].new_zeros(x[0].shape)
            tgt[:,:,1:] = x[0][:,:,:-1]
            tgt[:,:,0] = -1
            tgt_expected = src

            out, y_pred, mu_z, logvar_z  = self.model.forward(src,tgt)  # forward the data in the model

            sample_reconstructed = out[0].cpu().detach()
            batch_size = src.shape[0]
            out_reshape = out.contiguous().view(batch_size,-1)
            tgt_expected = tgt_expected.contiguous().view(batch_size,-1)

            loss = self.criterion(out_reshape,tgt_expected, mu_z, logvar_z, y_pred, y,i)
            return loss, sample_reconstructed, out


    def predict(self, x, y, i=1,max_length=128):
        self.eval()
        
        input_sequence = x[0]

        out = torch.tensor([[[0]]], dtype=torch.float, device=self.device).repeat(input_sequence.shape[0],input_sequence.shape[1],input_sequence.shape[2])
        # input_sequence [:,:,-1]==1
        src = x[0]
        tgt = x[0].new_zeros(x[0].shape)
        tgt[:,:,1:] = x[0][:,:,:-1]
        tgt[:,:,0] = -1
        tgt_expected = src
        
        out[:,:,0]=-1

# 
        with torch.no_grad():
            # for _ in range(max_length):
            out, y_pred, mu_z, logvar_z = self.model.generate(src.shape,tgt)
                # out, y_pred, mu_z, logvar_z = self.model(src,out)
       
        # print(out.shape)

        batch_size = input_sequence.shape[0]
        out_reshaped = out.contiguous().view(batch_size,-1)
        tgt_expected = input_sequence.contiguous().view(batch_size,-1)
        loss = self.criterion(out_reshaped,tgt_expected, mu_z, logvar_z, y_pred, y,i)

        # y_pred_probs = F.softmax(y_pred.data, dim=1)
        # y_pred = torch.argmax(y_pred_probs,1)
        # for i in torch.arange(src.shape[0]):
        #     print(y_pred[i:i+3])
        #     print(y[i:i+3])
        #     l = x[1][i]
        #     print(l)
        #     plt.scatter(range(128)[-l+1:],out[i,0,:].cpu()[-l+1:], label='predicted _{}'.format(y_pred[i].cpu()))
        #     plt.scatter(range(128)[-l+1:],input_sequence[i,0,:].cpu()[-l+1:], label='original _{}'.format(y[i].cpu()))
        #     plt.legend()
        #     plt.gca().invert_yaxis()
        #     plt.show()

        return loss, out

    # def generate(self, x, y, i=1,max_length=128):
    #     self.eval()
        
    #     src = x[0]
    #     tgt = x[0].new_zeros(x[0].shape)
    #     tgt[:,:,1:] = x[0][:,:,:-1]
    #     tgt[:,:,0] = -1
    #     tgt_expected = src

    #     with torch.no_grad():
    #         out, y_pred, mu_z, logvar_z = self.model.generate(tgt.shape, tgt)#, tgt_mask)

    #     print(out.shape)

    #     batch_size = tgt.shape[0]
    #     out_reshaped = out.contiguous().view(batch_size,-1)
    #     tgt_expected = tgt.contiguous().view(batch_size,-1)
    #     loss = self.criterion(out_reshaped,tgt_expected, mu_z, logvar_z, y_pred, y,i)

    #     y_pred_probs = F.softmax(y_pred.data, dim=1)
    #     y_pred = torch.argmax(y_pred_probs,1)
    #     print(y_pred)
    #     print(y)
    #     plt.scatter(range(128),out[0,1,:].cpu(), label='predicted')
    #     plt.scatter(range(128),tgt[0,1,:].cpu(), label='original')
    #     plt.legend()
    #     plt.gca().invert_yaxis()
    #     plt.show()

    #     return loss, out

    # def generate(self, x_shape=torch.Size([10,4,128]),i=1):
    #     self.eval()
        
    #     out = torch.tensor([[[0]]], dtype=torch.float, device=self.device).repeat(x_shape[0],x_shape[1],x_shape[2])
    #     out[:,:,0]=-1

    #     with torch.no_grad():
    #         for _ in range(x_shape[2]):
    #             out, y_pred, mu_z, logvar_z = self.model.generate(x_shape, out)

    #     y_pred_probs = F.softmax(y_pred.data, dim=1)
    #     y_pred = torch.argmax(y_pred_probs,1)

    #     return out, y_pred, y_pred_probs

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
        model_save_name="final_model.pth.tar", 
        data_name = 'final_training',
        train_data_name=''
        ):

        data_name=train_data_name+'final_training'

        start_time = time.time()
        n_epochs = n_epochs if n_epochs else self.best_epoch
        data_loaders = data_loaders if data_loaders else [self.train_data,self.val_data]

        if not self.pickup:
            print("resetting parameters...")
            self.model.reset_parameters()

        n_epochs=int(n_epochs)
        train_stats = torch.full((n_epochs+1,2),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch

        n_batches = sum([len(data) for data in data_loaders])

        with tqdm.tqdm(total=n_epochs) as pbar_train:

            for i, epoch_idx in enumerate(range(n_epochs)):
                
                epoch_start_time = time.time()
                running_loss = 0.0
                cm_idx = 0    

                for i, data in enumerate(data_loaders):
                    for idx, (x, y, ids) in enumerate(data):
                        loss = self.run_train_iter(x=x, y=y,i=epoch_idx)
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
        self.save_model(model_save_name) 
        
    def run_train_phase(self, model_save_name='best_validation_model.pth.tar', 
        model_load_name='best_validation_model.pth.tar',
        load_model=False,train_data_name=''):

        if load_model:
            self.load_model(model_save_name=model_load_name)

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
                    loss = self.run_train_iter(x=x, y=y,i=i)
                    
                    running_train_loss += loss.item()

                train_loss = running_train_loss/train_n_batches
                train_stats[epoch_idx,0] = epoch_idx+1 #epoch
                train_stats[epoch_idx,1] = train_loss #loss
            
                if step_count == self.validation_step:

                    step_count = 0
                    running_val_loss = 0.0

                    for idx, (x, y, ids) in enumerate(self.val_data):


                        # loss = self.run_evaluation_iter(x=x, y=y, i=idx)
                        loss, random_sample, out = self.run_evaluation_iter(x=x, y=y,i=i)
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
        plot_utils.plot_reconstructions_per_epoch(random_samples, self.validation_step, self.best_epoch, self.experiment_logs+'/'+train_data_name+"reconstruction_per_epoch.png")
        self.save_model(model_save_name)
        self.save_statistics(train_stats, train_data_name+'training_summary.csv')
        self.save_statistics(val_stats, train_data_name+'validation_summary.csv')
    
    
    def run_prediction(self, data=None, model_name="final_model.pth.tar",
        data_name="predicted",load_model=True):
        
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
                loss, out = self.predict(x=x,y=y,max_length=max_lc_length)
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

        preprocess_data_utils.save_vectors(results_dataset,self.parent.experiment_logs+'/'+'{}.h5'.format(data_name))
        self.save_statistics(test_stats, data_name+"_summary.csv")

    # def run_generation(self, model_name="final_model.pth.tar",
    #     data_name="fake_data",load_model=True, x_shape=torch.Size([64,4,128]), n_batches=10):
        
    #     start_time = time.time()
    #     if load_model:
    #         print("loading model")
    #         self.load_model(model_save_name=model_name)

    #     gen_ds_len = n_batches*x_shape[0]
    #     gen_x = torch.zeros(gen_ds_len,x_shape[1],x_shape[2])
    #     gen_lens = torch.full((gen_ds_len),x_shape[2])
    #     gen_y = torch.zeros(gen_ds_len)
    #     gen_ids = torch.arange(gen_ds_len)
    #     gen_y_probs = torch.zeros((gen_ds_len,self.num_output_classes), dtype=torch.double) # holds ids, probability predictions
        
              
    #     with tqdm.tqdm(total=n_batches) as pbar:
    #         for i in torch.arange(n_batches):
    #             out, y_pred, y_pred_probs = self.generate(x_shape=x_shape)
    #             gen_x[i*self.batch_size:i*self.batch_size+out.shape[0]] = out.detach().cpu() 
    #             gen_y[i*self.batch_size:i*self.batch_size+out.shape[0]] = y_pred.cpu()
    #             gen_y_probs[i*self.batch_size:i*self.batch_size+out.shape[0]] = y_pred_probs.detach().cpu()
    #             pbar.update(1)
    #             elapsed_time = time.time() - start_time 
    #             pbar.set_description("Batch: {:.3f}        , ET {:.2f}s".format(n_batches,elapsed_time))

    #     gen_probs=torch.cat((torch.unsqueeze(gen_ids,1),gen_y_probs), dim=1).cpu()

    #     results_dataset = {
    #         'X': gen_x,
    #         'Y': gen_y,
    #         'ids': gen_ids,
    #         'lens': gen_lens
    #     }

    #     preprocess_data_utils.save_vectors(results_dataset,self.parent.experiment_logs+'/'+'{}.h5'.format(data_name))
    #     self.save_probabilities(gen_probs, data_name+"_class_probabilities.csv")

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        self.parent.run_experiment()