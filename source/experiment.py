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
        continue_from_epoch=-1, 
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
        self.starting_epoch = 0

        if self.validation_step > self.num_epochs:
            print("Validation step should be less than the number of epochs so at least one run is possible")
            sys.exit()
        
        if self.patience > int(self.num_epochs/self.validation_step):
            print("Infinite patience, early stopping won't be an option")
        
        if train_data:
            if train_sampler is not None:
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
            else:
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            self.train_data = train_loader

        if val_data:
            if val_sampler is not None:
                val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size,sampler=val_sampler)
            else:
                val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
            self.val_data = val_loader

        if test_data:
            if test_sampler is not None:
                test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,sampler=test_sampler)
            else:
                test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)
            self.test_data = test_loader

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=False,
                                    weight_decay=weight_decay_coefficient)
        if class_weights:
            class_weights = torch.FloatTensor(class_weights).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)  # send the loss computation to the GPU

        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        if continue_from_epoch != -1:  # if continue from epoch is not -1 then
            try:
                self.best_val_model_idx, self.best_val_model_f1 = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="epoch",
                    model_idx=continue_from_epoch)  # reload existing model from epoch
                self.starting_epoch = continue_from_epoch
            except:
                print("Did not save that epoch's model, will start from 0")
        # Set best model f1_score to be at 0 and the best epoch to be num_epochs, since we are just starting
        self.best_val_model_idx = num_epochs
        self.best_val_model_f1 = 0

    def run_train_iter(self, x, y):
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
        predicted = F.softmax(out.data,dim=1)
        predicted = torch.argmax(predicted, 1)
        return loss, predicted

    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx):
        state = dict()
        state['network'] = self.model.state_dict()  # save network parameter and other variables.
        state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))

    def load_model(self, model_save_dir, model_save_name, model_idx):
        print(model_save_dir)
        print(model_save_name)
        print(model_idx)
        filename = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
        print(filename)
        state = torch.load(f=filename)
        self.model.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx']#, state['best_val_model_acc'], state['best_val_model_f1']

    def save_statistics(self, stats, fn):
        stats_keys = ['epoch', 'accuracy', 'loss', 'f1', 'precision', 'recall']
        stats_df = pd.DataFrame(stats.cpu().numpy() , columns = stats_keys)
        stats_df = stats_df[stats_df.epoch>=0]
        stats_df.epoch = stats_df.epoch.astype(int)
        stats_df.to_csv(self.experiment_logs+'/'+fn ,sep=',',index=False)

    def save_results(self, results, fn):
        results_keys = ['object_id','prediction','target']
        results_df = pd.DataFrame(results.cpu().numpy(), columns=results_keys)
        results_df.to_csv(self.experiment_logs+'/'+fn,sep=',',index=False)

    
    def run_test_phase(self, data=None, model_idx=None, model_name="final_model"):
        start_time = time.time()
        model_idx = model_idx if model_idx else self.best_val_model_idx
        data = data if data else self.test_data
        self.load_model(model_save_dir=self.experiment_saved_models, model_save_name=model_name,model_idx=model_idx)
        results_cm = torch.zeros((len(data.dataset),3), dtype=torch.int, device = self.device) # holds ids, preds, targets
        running_loss = 0.0
        
        with tqdm.tqdm(total=len(data)) as pbar:
            for i,(x, y, ids) in enumerate(data):
                loss, preds = self.run_evaluation_iter(x=x,y=y)
                torch.stack((ids,preds,y),dim=1,out=results_cm[i*self.batch_size])
                running_loss += loss.item()
                pbar.update(1)
                elapsed_time = time.time() - start_time 
                pbar.set_description("Test loss: {:.3f}        , ET {:.2f}s".format(running_loss/(i+1),elapsed_time))


        preds, targets = results_cm[:,1],results_cm[:,2]
        precision, recall = precision_recall(preds, targets, num_classes=self.num_output_classes, average='macro')
        loss = running_loss/len(data)
        test_stats = torch.full((1,6),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch
        test_stats[0,0] = 0 #epoch
        test_stats[0,1] = accuracy(preds,targets, num_classes=self.num_output_classes, average='micro') #accuracy
        test_stats[0,2] = loss #loss
        test_stats[0,3] = f1_score(preds, targets, num_classes=self.num_output_classes, average='macro') #f1
        test_stats[0,4] = precision # precision
        test_stats[0,5] = recall #recall

        self.save_statistics(test_stats,"test_summary.csv")
        self.save_results(results_cm,"test_results.csv")

    def run_final_train_phase(self, n_epochs=None, model_name="final_model"):
        start_time = time.time()
        n_epochs = n_epochs if n_epochs else self.best_val_model_idx
        data_loaders = [self.train_data,self.val_data]
        self.model.reset_parameters()
        train_stats = torch.full((n_epochs+1,6),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch

        n_batches = sum([len(data) for data in data_loaders])
        data_length = sum([len(data.dataset) for data in data_loaders])
        last_train_cm = torch.zeros((data_length,3), dtype=torch.int, device = self.device) # holds ids, preds, targets

        with tqdm.tqdm(total=n_epochs+1) as pbar_train:

            for i, epoch_idx in enumerate(range(self.starting_epoch, n_epochs+1)):
                
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

                train_stats[epoch_idx,0] = epoch_idx #epoch
                train_stats[epoch_idx,1] = accuracy(preds,targets, num_classes=self.num_output_classes, average='micro') #accuracy
                train_stats[epoch_idx,2] = train_loss #loss
                train_stats[epoch_idx,3] = f1_score(preds, targets, num_classes=self.num_output_classes, average='macro') #f1
                train_stats[epoch_idx,4] = precision # precision
                train_stats[epoch_idx,5] = recall #recall

                pbar_train.update(1)
                elapsed_time = time.time() - start_time  
                pbar_train.set_description("Train loss: {:.3f}       , ET {:.2f}s".format(train_loss,elapsed_time))
                
        self.save_statistics(train_stats, 'final_training_summary.csv')
        self.save_results(last_train_cm, 'final_training_results.csv')
        self.save_model(model_save_dir=self.experiment_saved_models,
            model_save_name=model_name, model_idx=epoch_idx,best_validation_model_idx=self.best_val_model_idx) 
        
    def run_train_phase(self):
        start_time = time.time()

        strike = 0
        step_count = 0
        last_val_performance = 0

        last_train_cm = torch.zeros((len(self.train_data.dataset),3), dtype=torch.int, device=self.device) # holds ids, preds, targets
        current_val_cm = torch.zeros((len(self.val_data.dataset),3), dtype=torch.int, device=self.device) # holds ids, preds, targets
        best_val_cm = torch.zeros((len(self.val_data.dataset),3), dtype=torch.int, device=self.device) # holds ids, preds, targets
        
        val_stats = torch.full((int(self.num_epochs/self.validation_step),6),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per validation epoch
        train_stats = torch.full((int(self.num_epochs),6),-1, dtype=torch.float, device=self.device) # holds epoch, acc, loss, f1, precission, recall, per train epoch
        
        train_n_batches = len(self.train_data)
        val_n_batches = len(self.val_data)
        val_loss = -1
        val_idx = 0

        with tqdm.tqdm(total=self.num_epochs) as pbar_train:

            for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
                step_count+=1
                running_train_loss = 0.0

                for idx, (x, y,ids) in enumerate(self.train_data):
                    loss, preds = self.run_train_iter(x=x, y=y)
                    running_train_loss += loss.item()
                    torch.stack((ids,preds,y),dim=1,out=last_train_cm[idx*self.batch_size])
                
                preds, targets = last_train_cm[:,1],last_train_cm[:,2]
                precision, recall = precision_recall(preds,targets, num_classes=self.num_output_classes, average='macro')
                train_loss = running_train_loss/train_n_batches

                train_stats[epoch_idx,0] = epoch_idx #epoch
                train_stats[epoch_idx,1] = accuracy(preds,targets, num_classes=self.num_output_classes, average='micro') #accuracy
                train_stats[epoch_idx,2] = train_loss #loss
                train_stats[epoch_idx,3] = f1_score(preds,targets, num_classes = self.num_output_classes, average='macro') #f1
                train_stats[epoch_idx,4] = precision # precision
                train_stats[epoch_idx,5] = recall #recall
            
                if step_count == self.validation_step:

                    step_count = 0
                    running_val_loss = 0.0

                    for idx, (x, y, ids) in enumerate(self.val_data):
                        loss, preds = self.run_evaluation_iter(x=x, y=y)
                        running_val_loss += loss.item()
                        torch.stack((ids,preds,y),dim=1,out=current_val_cm[idx*self.batch_size])

                    preds, targets = current_val_cm[:,1],current_val_cm[:,2]
                    precision, recall = precision_recall(preds,targets, num_classes=self.num_output_classes, average='macro')
                    f1 = f1_score(preds,targets, num_classes=self.num_output_classes, average='macro')
                    val_loss = running_val_loss/val_n_batches

                    val_stats[val_idx,0] = epoch_idx #epoch
                    val_stats[val_idx,1] = accuracy(preds,targets, num_classes=self.num_output_classes, average='micro') #accuracy
                    val_stats[val_idx,2] = val_loss #loss
                    val_stats[val_idx,3] = f1 #f1
                    val_stats[val_idx,4] = precision # precision
                    val_stats[val_idx,5] = recall #recall
                    val_idx+=1

                    #if the f1-score of the current epoch for the validation set is better than previous epochs
                    if f1 > self.best_val_model_f1:
                        best_val_cm = current_val_cm
                        self.best_val_model_f1 = f1
                        self.best_val_model_idx = epoch_idx
                        self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="epoch", model_idx=epoch_idx,best_validation_model_idx=self.best_val_model_idx)
                        strike = 0

                    else:
                        strike+=1

                    last_val_performance = f1

                pbar_train.update(1)
                elapsed_time = time.time() - start_time 
                pbar_train.set_description("Tr/Val loss: {:.3f}/{:.3f}, Strike: {}, ET {:.2f}s".format(train_loss,val_loss,strike,elapsed_time))
                
                if strike == self.patience:
                    pbar_train.set_description("Tr/Val loss: {:.3f}/{:.3f}, Best epoch: {}, ET {:.2f}s".format(train_loss,val_loss,self.best_val_model_idx,elapsed_time))
                    break

        #save statistics at the end only
        self.save_statistics(train_stats, 'training_summary.csv')
        self.save_results(last_train_cm, 'training_results.csv')
        self.save_statistics(val_stats, 'validation_summary.csv')
        self.save_results(best_val_cm, 'validation_results.csv')
    
    
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
            print("")
            print("Starting final training phase")
            print("")
            self.run_final_train_phase()

        if self.test_data:
            print("")
            print("Starting test phase")
            print("")
            self.run_test_phase(self.test_data)
