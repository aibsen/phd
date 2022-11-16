from torch import nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import pandas as pd
import time
from torchmetrics.functional import precision_recall, f1_score, accuracy
# from utils import save_to_stats_pkl_file, load_from_stats_pkl_file, \
#     save_statistics, load_statistics, save_classification_results

class ClassificationExperiment():
    
    def __init__(self, parent):
        self.parent = parent

    def __getattr__(self, attr):
        return getattr(self.parent, attr)

    def run_train_iter(self, x, y):
        # print(x.shape)

        self.train()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        out = self.model.forward(x)  # forward the data in the model
        loss = self.criterion(out,y)
        loss.backward()  # backpropagate
        self.optimizer.step()
        predicted = F.softmax(out.data,dim=1)
        predicted = torch.argmax(predicted, 1)
        return loss, predicted

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode
        out = self.model.forward(x)  # forward the data in the model
        loss =  self.criterion(out,y)
        predicted_soft = F.softmax(out.data,dim=1)
        # predicted_soft = F.softmax(F.softmax(out.data,dim=1),dim=1)
        # print(predicted_soft)
        # print(predicted_soft.shape)
        predicted = torch.argmax(predicted_soft, 1)
        return loss, predicted, predicted_soft

    def save_model(self, model_save_name):
        self.parent.save_model(model_save_name)

    def load_model(self, model_save_name):
        self.parent.save_model(model_save_name)

    def save_statistics(self, stats, fn):
        stats_keys = ['epoch', 'accuracy', 'loss', 'f1', 'precision', 'recall']
        self.parent.save_statistics(stats,fn,stats_keys)
    
    def save_results(self, results, fn):
        results_keys = ['object_id','prediction','target']
        results_df = pd.DataFrame(results.cpu().numpy(), columns=results_keys)
        results_df.to_csv(self.parent.experiment_logs+'/'+fn,sep=',',index=False)

    def save_probabilities(self, results, fn):
        results_keys = ['object_id']+[str(i) for i in range (self.num_output_classes)]
        results_df = pd.DataFrame(results.cpu().numpy(), columns=results_keys)
        results_df['object_id'] = results_df['object_id'].astype('int64')
        results_df.to_csv(self.parent.experiment_logs+'/'+fn,sep=',',index=False)


    def run_test_phase(self, data=None, model_name="final_model.pth.tar",
        data_name="test",load_model=True):

        start_time = time.time()
        data = data if data else self.test_data
        if load_model:
            self.load_model(model_save_name=model_name)
        results_cm = torch.zeros((len(data.dataset),3), dtype=torch.int64, device = self.device) # holds ids, preds, targets
        results_probs = torch.zeros((len(data.dataset),self.num_output_classes+1), dtype=torch.double, device = self.device) # holds ids, probability predictions
        running_loss = 0.0
        
        with tqdm.tqdm(total=len(data)) as pbar:
            for i,(x, y, ids) in enumerate(data):
                loss, preds, preds_soft = self.run_evaluation_iter(x=x,y=y)
                torch.stack((ids,preds,y),dim=1,out=results_cm[i*self.batch_size])
                torch.cat((torch.unsqueeze(ids,1),preds_soft), dim=1, out=results_probs[i*self.batch_size])
                running_loss += loss.item()
                pbar.update(1)
                elapsed_time = time.time() - start_time 
                pbar.set_description("Test loss: {:.3f}        , ET {:.2f}s".format(running_loss/(i+1),elapsed_time))


        preds, targets = results_cm[:,1],results_cm[:,2]
        precision, recall = precision_recall(preds, targets, num_classes=self.num_output_classes, average='macro')
        loss = running_loss/len(data)
        test_stats = torch.full((1,6),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch
        test_stats[0,0] = 1 #epoch
        test_stats[0,1] = accuracy(preds,targets, num_classes=self.num_output_classes, average='micro') #accuracy
        test_stats[0,2] = loss #loss
        test_stats[0,3] = f1_score(preds, targets, num_classes=self.num_output_classes, average='macro') #f1
        test_stats[0,4] = precision # precision
        test_stats[0,5] = recall #recall

        self.save_probabilities(results_probs, data_name+"_probabilities.csv")
        self.save_statistics(test_stats, data_name+"_summary.csv")
        self.save_results(results_cm, data_name+"_results.csv")

    def run_final_train_phase(self, data_loaders=None,n_epochs=None, 
        model_name="final_model.pth.tar", data_name = 'final_training'):

        start_time = time.time()
        n_epochs = n_epochs if n_epochs else self.best_epoch
        n_epochs = int(n_epochs)
        data_loaders = data_loaders if data_loaders else [self.train_data,self.val_data]
        self.model.reset_parameters()
        train_stats = torch.full((int(n_epochs+1),6),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch

        n_batches = sum([len(data) for data in data_loaders])
        data_length = sum([len(data.dataset) for data in data_loaders])
        last_train_cm = torch.zeros((data_length,3), dtype=torch.int64, device = self.device) # holds ids, preds, targets

        with tqdm.tqdm(total=n_epochs) as pbar_train:

            for i, epoch_idx in enumerate(range(n_epochs)):
                
                epoch_start_time = time.time()
                running_loss = 0.0
                cm_idx = 0    

                for i, data in enumerate(data_loaders):
                    for idx, (x, y, ids) in enumerate(data):
                        loss, preds = self.run_train_iter(x=x, y=y)
                        running_loss += loss.item()
                        batch_size = len(x)
                        torch.stack((ids,preds,y),dim=1,out=last_train_cm[cm_idx])
                        cm_idx+=batch_size

                preds, targets = last_train_cm[:,1],last_train_cm[:,2]
                precision, recall = precision_recall(preds, targets, num_classes=self.num_output_classes, average='macro')
                train_loss = running_loss/n_batches
                f1=f1_score(preds, targets, num_classes=self.num_output_classes, average='macro') #f1

                train_stats[epoch_idx,0] = epoch_idx+1 #epoch
                train_stats[epoch_idx,1] = accuracy(preds,targets, num_classes=self.num_output_classes, average='micro') #accuracy
                train_stats[epoch_idx,2] = train_loss #loss
                train_stats[epoch_idx,3] = f1
                train_stats[epoch_idx,4] = precision # precision
                train_stats[epoch_idx,5] = recall #recall

                pbar_train.update(1)
                elapsed_time = time.time() - start_time  
                pbar_train.set_description("Train loss: {:.3f}       , ET {:.2f}s".format(train_loss,elapsed_time))
                
        self.save_statistics(train_stats, '{}_summary.csv'.format(data_name))
        self.save_results(last_train_cm, '{}_results.csv'.format(data_name))
        self.state = {
            'epoch': n_epochs,
            'model': self.model.state_dict(),
            'optimizer':self.optimizer.state_dict()
                    }
        self.save_model(model_name) 
        
    def run_train_phase(self, model_name='best_validation_model.pth.tar', load_model=False):

        start_time = time.time()
        strike = 0
        step_count = 0
        last_val_performance = 0

        last_train_cm = torch.zeros((len(self.train_data.dataset),3), dtype=torch.int64, device=self.device) # holds ids, preds, targets
        current_val_cm = torch.zeros((len(self.val_data.dataset),3), dtype=torch.int64, device=self.device) # holds ids, preds, targets
        best_val_cm = torch.zeros((len(self.val_data.dataset),3), dtype=torch.int64, device=self.device) # holds ids, preds, targets
        
        val_stats = torch.full((int(self.num_epochs/self.validation_step),6),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per validation epoch
        train_stats = torch.full((int(self.num_epochs),6),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch
        
        train_n_batches = len(self.train_data)
        val_n_batches = len(self.val_data)
        val_loss = -1
        val_idx = 0

        with tqdm.tqdm(total=self.num_epochs) as pbar_train:

            for i, epoch_idx in enumerate(range(self.num_epochs)):
                step_count+=1
                running_train_loss = 0.0

                for idx, (x, y,ids) in enumerate(self.train_data):
                    loss, preds = self.run_train_iter(x=x, y=y)
                    running_train_loss += loss.item()
                    torch.stack((ids,preds,y),dim=1,out=last_train_cm[idx*self.batch_size])

                preds, targets = last_train_cm[:,1],last_train_cm[:,2]
                precision, recall = precision_recall(preds,targets, num_classes=self.num_output_classes, average='macro')
                train_loss = running_train_loss/train_n_batches

                train_stats[epoch_idx,0] = epoch_idx+1 #epoch
                train_stats[epoch_idx,1] = accuracy(preds,targets, num_classes=self.num_output_classes, average='micro') #accuracy
                train_stats[epoch_idx,2] = train_loss #loss
                train_stats[epoch_idx,3] = f1_score(preds,targets, num_classes = self.num_output_classes, average='macro') #f1
                train_stats[epoch_idx,4] = precision # precision
                train_stats[epoch_idx,5] = recall #recall
            
                if step_count == self.validation_step:

                    step_count = 0
                    running_val_loss = 0.0

                    for idx, (x, y, ids) in enumerate(self.val_data):
                        loss, preds, _ = self.run_evaluation_iter(x=x, y=y)
                        running_val_loss += loss.item()
                        torch.stack((ids,preds,y),dim=1,out=current_val_cm[idx*self.batch_size])

                    preds, targets = current_val_cm[:,1],current_val_cm[:,2]
                    precision, recall = precision_recall(preds,targets, num_classes=self.num_output_classes, average='macro')
                    f1 = f1_score(preds,targets, num_classes=self.num_output_classes, average='macro')
                    val_loss = running_val_loss/val_n_batches

                    val_stats[val_idx,0] = epoch_idx+1 #epoch
                    val_stats[val_idx,1] = accuracy(preds,targets, num_classes=self.num_output_classes, average='micro') #accuracy
                    val_stats[val_idx,2] = val_loss #loss
                    val_stats[val_idx,3] = f1 #f1
                    val_stats[val_idx,4] = precision # precision
                    val_stats[val_idx,5] = recall #recall
                    val_idx+=1

                    #if the f1-score of the current epoch for the validation set is better than previous epochs
                    if f1 > self.best_f1:
                        best_val_cm = current_val_cm
                        self.best_f1 = f1
                        self.best_epoch = epoch_idx
                        self.state = {
                            'epoch': epoch_idx,
                            'model': self.model.state_dict(),
                            'optimizer':self.optimizer.state_dict()
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
        self.save_results(last_train_cm, 'training_results.csv')
        self.save_statistics(val_stats, 'validation_summary.csv')
        self.save_results(best_val_cm, 'validation_results.csv')
    
    
    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        self.parent.run_experiment()
