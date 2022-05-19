from torch import nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import tqdm
import os
import numpy as np
import pandas as pd
import time
from torch.utils.data import SequentialSampler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torchmetrics import Accuracy, F1Score
from torchmetrics.functional import precision_recall
# from utils import save_to_stats_pkl_file, load_from_stats_pkl_file, \
#     save_statistics, load_statistics, save_classification_results


class Experiment(nn.Module):
    def __init__(self, network_model, 
        experiment_name,metric="f1_score", 
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
        best_idx=0, 
        cached_dataset=False,
        patience=3,
        validation_step=5,
        class_weights = None):

        super(Experiment, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # print("using GPU")

        self.metric = metric
        self.experiment_name = experiment_name
        self.model = network_model
        self.model.to(self.device)
        self.model.reset_parameters()
        self.patience = patience
        self.validation_step = validation_step
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.break_epoch = num_epochs
        self.num_output_classes = num_output_classes

        self.train_data=None
        self.val_data=None
        self.test_data=None
        self.starting_epoch = 0
        
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

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = best_idx
        self.best_val_model_f1 = 0

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

        self.accuracy = Accuracy(num_classes = self.num_output_classes, average = 'micro')
        self.f1 = F1Score(num_classes = self.num_output_classes, average='macro')

    def run_train_iter(self, x, y):
        self.train()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        out = self.model.forward(x)  # forward the data in the model
        loss = self.criterion(out,y)
        loss.backward()  # backpropagate
        self.optimizer.step()
        predicted = F.log_softmax(out.data,dim=1)
        predicted = torch.argmax(predicted, 1)
        return loss, predicted

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode
        out = self.model.forward(x)  # forward the data in the model
        loss =  self.criterion(out,y)
        predicted = F.log_softmax(out.data,dim=1)
        predicted = torch.argmax(predicted, 1)
        return loss, predicted

    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx):
        state = dict()
        state['network'] = self.model.state_dict()  # save network parameter and other variables.
        state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))

    def load_model(self, model_save_dir, model_save_name, model_idx):
        filename = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
        state = torch.load(f=filename)
        self.model.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx']#, state['best_val_model_acc'], state['best_val_model_f1']

    def save_statistics(self, experiment_log_dir, filename, stats_dict):
        df = pd.DataFrame(stats_dict)
        path = os.path.join(experiment_log_dir, filename)
        df.to_csv(path,sep=',',index=False)

    def run_test_phase(self, data, model_idx, experiment_log_dir, model_name="final_model"):
        start_time = time.time()
        self.load_model(model_save_dir=experiment_log_dir, model_save_name=model_name,model_idx=model_idx)

        test_targets=[]
        test_predictions=[]
        test_loss_cum=0
        test_ids=[]

        with tqdm.tqdm(total=len(data)) as pbar:
            for i,(x, y, ids) in enumerate(data):
                loss, pred, targets = self.run_evaluation_iter(x=x,y=y)
                test_ids+=list(ids.cpu().numpy())
                test_loss_cum += loss
                test_predictions += pred
                test_targets += targets
                pbar.update(1)
                elapsed_time = time.time() - start_time 
                pbar.set_description("Test loss: {:.3f}        , ET {:.2f}s".format(test_loss_cum/(i+1),elapsed_time))

        loss = test_loss_cum/len(data)
        accuracy = accuracy_score(test_targets, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average = 'macro', zero_division=0)

        metrics={"test_acc": [accuracy], "test_loss": [loss], "test_f1": [f1],"test_precision":[precision],"test_recall":[recall]}
        results = pd.DataFrame({'ids':test_ids,'targets':test_targets,'predictions':test_predictions }) 

        self.save_statistics(experiment_log_dir=experiment_log_dir,filename='test_results.csv',stats_dict=results)
        self.save_statistics(experiment_log_dir=experiment_log_dir, filename="test_summary.csv",stats_dict=metrics)

    def run_final_train_phase(self, data_loaders, experiment_log_dir, model_name="final_model"):
        start_time = time.time()
                
        self.model.reset_parameters()
        train_stats = {"epoch": [],"train_acc": [], "train_loss": [], "train_f1":[],"train_precision":[],"train_recall":[]}  # initialize a dict to keep the per-epoch metrics

        n_batches = sum([len(data) for data in data_loaders])
        with tqdm.tqdm(total=self.break_epoch) as pbar_train:

            for i, epoch_idx in enumerate(range(self.starting_epoch, self.break_epoch)):
                
                epoch_start_time = time.time()
                train_loss_cum = 0 
                train_predictions = []
                train_targets = []

                for data in data_loaders:
                    for idx, (x, y,ids) in enumerate(data):
                        loss, pred, targets = self.run_train_iter(x=x, y=y)
                        train_loss_cum += loss
                        train_predictions += pred
                        train_targets += targets
                        
                
                loss = train_loss_cum/n_batches
                accuracy = accuracy_score(train_targets, train_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(train_targets, train_predictions, average = 'macro', zero_division=0)
                
                train_stats['train_acc'].append(accuracy)
                train_stats['train_loss'].append(loss)
                train_stats['train_f1'].append(f1)
                train_stats['train_recall'].append(recall)
                train_stats['train_precision'].append(precision)
                train_stats['epoch'].append(epoch_idx)
                pbar_train.update(1)

                elapsed_time = time.time() - start_time  
                pbar_train.set_description("Train loss: {:.3f}       , ET {:.2F}s".format(loss,elapsed_time))
                
                
        self.save_statistics(experiment_log_dir=experiment_log_dir, filename='final_training_summary.csv',
            stats_dict=train_stats)  # save statistics
        self.save_model(model_save_dir=experiment_log_dir,
                    model_save_name=model_name, model_idx=self.break_epoch,best_validation_model_idx=self.break_epoch)    
        
    def run_train_phase(self):
        start_time = time.time()

        strike = 0
        step_count = 0
        last_val_performance = 0
        last_train_cm = torch.zeros((len(self.train_data.dataset),3), dtype=torch.int) # holds ids, preds, targets
        current_val_cm = torch.zeros((len(self.val_data.dataset),3), dtype=torch.int) # holds ids, preds, targets
        best_val_cm = torch.zeros((len(self.val_data.dataset),3), dtype=torch.int) # holds ids, preds, targets
        val_stats = torch.full((int(self.num_epochs/self.validation_step),6),-1, dtype=torch.float) # holds epoch, acc, loss, f1, precission, recall, per validation epoch
        train_stats = torch.full((int(self.num_epochs),6),-1, dtype=torch.float) # holds epoch, acc, loss, f1, precission, recall, per train epoch
        val_idx = 0
        train_n_batches = len(self.train_data)
        val_n_batches = len(self.val_data)
        val_loss = -1


        with tqdm.tqdm(total=self.num_epochs) as pbar_train:

            for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
                
                step_count+=1
                running_train_loss = 0.0

                for idx, (x, y,ids) in enumerate(self.train_data):
                    loss, pred = self.run_train_iter(x=x, y=y)
                    last_train_cm[idx*self.batch_size:(idx+1)*self.batch_size,0] = ids
                    last_train_cm[idx*self.batch_size:(idx+1)*self.batch_size,1] = pred
                    last_train_cm[idx*self.batch_size:(idx+1)*self.batch_size,2] = y
                    running_train_loss += loss.item()
                
                preds, targets = last_train_cm[:,1],last_train_cm[:,2]
                precision, recall = precision_recall(preds,targets, num_classes=self.num_output_classes, average='macro')
                f1 = self.f1(preds,targets)
                train_loss = running_train_loss/train_n_batches
                train_stats[epoch_idx,0] = epoch_idx #epoch
                train_stats[epoch_idx,1] = self.accuracy(preds,targets) #accuracy
                train_stats[epoch_idx,2] = train_loss #loss
                train_stats[epoch_idx,3] = f1 #f1
                train_stats[epoch_idx,4] = precision # precision
                train_stats[epoch_idx,5] = recall #recall
            
                if step_count == self.validation_step:

                    step_count = 0
                    running_val_loss = 0.0

                    for idx, (x, y, ids) in enumerate(self.val_data):
                        loss, pred = self.run_evaluation_iter(x=x, y=y)
                        current_val_cm[idx*self.batch_size:(idx+1)*self.batch_size,0] = ids
                        current_val_cm[idx*self.batch_size:(idx+1)*self.batch_size,1] = pred
                        current_val_cm[idx*self.batch_size:(idx+1)*self.batch_size,2] = y
                        running_val_loss += loss.item()

                    preds, targets = current_val_cm[:,1],current_val_cm[:,2]
                    precision, recall = precision_recall(preds,targets, num_classes=self.num_output_classes, average='macro')
                    f1 = self.f1(preds,targets)
                    val_loss = running_val_loss/val_n_batches
                    val_stats[val_idx,0] = epoch_idx #epoch
                    val_stats[val_idx,1] = self.accuracy(preds,targets) #accuracy
                    val_stats[val_idx,2] = val_loss #loss
                    val_stats[val_idx,3] = f1 #f1
                    val_stats[val_idx,4] = precision # precision
                    val_stats[val_idx,5] = recall #recall
                    val_idx+=1
                    # print(val_stats)
                    # print(f1)
                #if the f1-score of the current epoch for the validation set is better than previous epochs
                    if f1 > self.best_val_model_f1:
                        best_val_cm = current_val_cm
                        self.best_val_model_f1 = f1
                        self.best_val_model_idx = epoch_idx
                        self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="epoch", model_idx=epoch_idx,best_validation_model_idx=self.best_val_model_idx)
                        self.break_epoch = epoch_idx
                        strike = 0
                        # print(f1.item())

                    elif f1 < last_val_performance:
                        strike+=1
                        # print(f1.item())
                        # print("strike {} at ep {}".format(strike, i))
                    last_val_performance = f1

                pbar_train.update(1)
                elapsed_time = time.time() - start_time 
                pbar_train.set_description("Tr/Val loss: {:.3f}/{:.3f}, Strike: {}, ET {:.2f}s".format(train_loss,val_loss,strike,elapsed_time))
                # pbar_train.set_description("Strikes: {}, ET {:.2f}s".format(strike,elapsed_time))

                if strike == self.patience:
                    # print("Early stop, model is overfitting")
                    break

        #save statistics at the end only
        self.save_train_val_statistics0(train_stats, val_stats, last_train_cm, best_val_cm)
    
    def save_train_val_statistics0(self, train_stats, val_stats, train_last_cm, val_last_cm):
        
        stats_keys = ['epoch', 'accuracy', 'loss', 'f1', 'precision', 'recall']
        train_stats = train_stats.cpu().numpy()
        val_stats = val_stats.cpu().numpy()
        train_df = pd.DataFrame(train_stats, columns = stats_keys)
        train_df.epoch = train_df.epoch.astype(int)
        val_df = pd.DataFrame(val_stats, columns = stats_keys)
        val_df.epoch = val_df.epoch.astype(int)
        train_df.to_csv(self.experiment_logs+'/training_summary.csv',sep=',',index=False)
        val_df.to_csv(self.experiment_logs+'/validation_summary.csv',sep=',',index=False)

        results_keys = ['object_id','prediction','target']
        train_cm = train_last_cm.cpu().numpy()
        val_cm = val_last_cm.cpu().numpy()
        train_cm_df = pd.DataFrame(train_cm, columns=results_keys)
        val_cm_df = pd.DataFrame(val_cm, columns=results_keys)
        train_cm_df.to_csv(self.experiment_logs+'/train_results.csv',sep=',',index=False)
        val_cm_df.to_csv(self.experiment_logs+'/validation_results.csv',sep=',',index=False)

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
            self.run_final_train_phase([self.train_data,self.val_data],self.experiment_logs)

        if self.test_data:
            print("")
            print("Starting test phase")
            print("")
            self.run_test_phase(self.test_data, self.break_epoch,self.experiment_logs)
