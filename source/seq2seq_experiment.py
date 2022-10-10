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
import sys
from torchmetrics.functional import precision_recall, f1_score, accuracy
# from utils import save_to_stats_pkl_file, load_from_stats_pkl_file, \
#     save_statistics, load_statistics, save_classification_results


class Experiment(nn.Module):
    def __init__(self, network_model, 
        experiment_name,
        num_epochs=100,
        num_output_classes=14, 
        learning_rate=1e-03,
        batch_size = 64, 
        train_data=None, 
        val_data=None,
        test_data=None,
        train_sampler=None,
        val_sampler=None,
        test_sampler = None,
        weight_decay_coefficient=1e-03, 
        patience=3,
        validation_step=3,
        class_weights = None):

        super(Experiment, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # print("using GPU")

        self.experiment_name = experiment_name
        self.model = network_model
        self.model.to(self.device)
        self.model.reset_parameters()
        self.patience = patience
        self.validation_step = validation_step
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_output_classes = num_output_classes

        self.train_data=None
        self.val_data=None
        self.test_data=None

        if self.validation_step > self.num_epochs:
            print("Validation step should be less than the number of epochs so at least one run is possible")
            sys.exit()
        
        if self.patience > int(self.num_epochs/self.validation_step):
            print("Infinite patience, early stopping won't be an option")
        
        if train_data:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            self.train_data = train_loader

        if val_data:
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
            self.val_data = val_loader

        if test_data:
            test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)
            self.test_data = test_loader

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=False,
                                    weight_decay=weight_decay_coefficient)
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU

        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        # Set best model f1_score to be at 0 and the best epoch to be num_epochs, since we are just starting
        self.best_epoch = num_epochs
        self.best_loss = np.inf
        self.state = {}

    def run_train_iter(self, x, y):
        self.train()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        src = x[0].permute(0,2,1)
        src = (src,x[1])
        out = self.model.forward(src,src)  # forward the data in the model
        # out = self.model.forward(x)
        # print(out.shape)
        # print(src[0].shape)
        loss = self.criterion(out,y)
        # loss = self.criterion(out,src[0])
        loss.backward()  # backpropagate
        self.optimizer.step()
        return loss


        #     # print(x)
#     print(x[0].shape)
#     src = x[0].permute(0,2,1)
#     src = (src,x[1])
#     a =nn(src,src)
#     print(a.shape)

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode
        src = x[0].permute(0,2,1)
        src = (src,x[1])
        out = self.model.forward(src,src)  # forward the data in the model
        loss = self.criterion(out,y)
        # loss = self.criterion(out,src[0])
        return loss

    def save_model(self, model_save_name):
        save_path = os.path.join(self.experiment_saved_models, model_save_name)
        torch.save(self.state,save_path)

    def load_model(self, model_save_dir, model_save_name):
        save_path = os.path.join(model_save_dir, model_save_name)
        self.state = torch.load(f=save_path)
        self.model.load_state_dict(state_dict=self.state['model'])
        self.optimizer.load_state_dict(state_dict=self.state['optimizer'])
        self.best_epoch = self.state['epoch']
        self.best_f1 = self.state['f1']
        # return state['best_val_model_idx']#, state['best_val_model_acc'], state['best_val_model_f1']

    def save_statistics(self, stats, fn):
        stats_keys = ['epoch', 'loss']
        stats_df = pd.DataFrame(stats.cpu().numpy() , columns = stats_keys)
        stats_df = stats_df[stats_df.epoch>=0]
        stats_df.epoch = stats_df.epoch.astype(int)
        stats_df.to_csv(self.experiment_logs+'/'+fn ,sep=',',index=False)
    
    def save_results(self, results, fn):
        results_keys = ['object_id','prediction','target']
        results_df = pd.DataFrame(results.cpu().numpy(), columns=results_keys)
        results_df.to_csv(self.experiment_logs+'/'+fn,sep=',',index=False)

    def save_probabilities(self, results, fn):
        results_keys = ['object_id']+[str(i) for i in range (self.num_output_classes)]
        results_df = pd.DataFrame(results.cpu().numpy(), columns=results_keys)
        results_df['object_id'] = results_df['object_id'].astype('int64')
        results_df.to_csv(self.experiment_logs+'/'+fn,sep=',',index=False)


    def run_test_phase(self, data=None, model_name="final_model.pth.tar",data_name="test",load_model=True):
        start_time = time.time()
        data = data if data else self.test_data
        if load_model:
            self.load_model(model_save_dir=self.experiment_saved_models, model_save_name=model_name)
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

    def run_final_train_phase(self, data_loaders=None,n_epochs=None, model_name="final_model.pth.tar", data_name = 'final_training'):
        print(self.num_output_classes)
        start_time = time.time()
        n_epochs = n_epochs if n_epochs else self.best_epoch
        data_loaders = data_loaders if data_loaders else [self.train_data,self.val_data]
        self.model.reset_parameters()
        train_stats = torch.full((n_epochs+1,6),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch

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
            'optimizer':self.optimizer.state_dict(),
            'f1':f1
                    }
        self.save_model(model_name) 
        
    def run_train_phase(self):
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
                    loss = self.run_train_iter(x=x, y=y)
                    
                    running_train_loss += loss.item()

                train_loss = running_train_loss/train_n_batches
                train_stats[epoch_idx,0] = epoch_idx+1 #epoch
                train_stats[epoch_idx,1] = train_loss #loss
            
                if step_count == self.validation_step:

                    step_count = 0
                    running_val_loss = 0.0

                    for idx, (x, y, ids) in enumerate(self.val_data):
                        loss = self.run_evaluation_iter(x=x, y=y)
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
        if self.train_data and self.val_data:
            print("")
            print("Starting training phase")
            print("")
            self.run_train_phase()
        #     print("")
        #     print("Starting final training phase")
        #     print("")
        #     self.run_final_train_phase()

        # if self.test_data:
        #     print("")
        #     print("Starting test phase")
        #     print("")
        #     self.run_test_phase(self.test_data)
