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
# from utils import save_to_stats_pkl_file, load_from_stats_pkl_file, \
#     save_statistics, load_statistics, save_classification_results


class Experiment(nn.Module):
    def __init__(self, network_model, 
        experiment_name,metric="f1_score", 
        num_epochs=100, 
        learning_rate=1e-03,
        batch_size = 64, 
        train_data=None, 
        val_data=None,
        test_data=None,
        train_sampler=None,
        val_sampler=None,
        test_sampler = None,
        weight_decay_coefficient=1e-03, 
        use_gpu=True, 
        continue_from_epoch=-1, 
        best_idx=0, 
        verbose=True,
        cached_dataset=False,
        patience=5,
        validation_step=5,
        class_weights = None):

        super(Experiment, self).__init__()

        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device('cuda')
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            if verbose:
                print("using GPU")
        else:
            if verbose:
                print("using CPU")
            self.device = torch.device('cpu')

        self.metric = metric
        self.verbose = verbose
        self.experiment_name = experiment_name
        self.model = network_model
        self.model.to(self.device)
        self.model.reset_parameters()
        self.patience = patience
        self.validation_step = validation_step

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

        self.num_epochs = num_epochs
        self.break_epoch = num_epochs
        if class_weights:
            class_weights = torch.FloatTensor(class_weights).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)  # send the loss computation to the GPU

        if continue_from_epoch != -1:  # if continue from epoch is not -1 then
            try:
                self.best_val_model_idx, self.best_val_model_f1 = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="epoch",
                    model_idx=continue_from_epoch)  # reload existing model from epoch
                self.starting_epoch = continue_from_epoch
            except:
                print("Did not save that epoch's model, will start from 0")

    def run_train_iter(self, x, y):
        self.train()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        out = self.model.forward(x)  # forward the data in the model
        loss = self.criterion(out,y)
        loss.backward()  # backpropagate
        self.optimizer.step()
        predicted = F.log_softmax(out.data,dim=1)
        predicted = torch.argmax(predicted, 1)
        loss = loss.data.cpu().numpy()
        y_cpu = list(y.data.cpu().numpy())
        predicted_cpu = list(predicted.cpu().numpy())
        return loss, predicted_cpu, y_cpu

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode
        out = self.model.forward(x)  # forward the data in the model
        loss =  self.criterion(out,y)
        predicted = F.log_softmax(out.data,dim=1)
        predicted = torch.argmax(predicted, 1)
        loss = loss.data.cpu().numpy()
        y_cpu = list(y.data.cpu().numpy())
        predicted_cpu = list(predicted.cpu().numpy())
        return loss, predicted_cpu, y_cpu

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
        train_stats = {"epoch": [],"train_acc": [], "train_loss": [], "train_f1":[],"train_precision":[],"train_recall":[]}  # initialize a dict to keep the per-epoch metrics
        val_stats = {"epoch": [],"val_acc": [],"val_loss": [], "val_f1":[], "val_precision":[],"val_recall":[]}
        
        strike = 0
        step_count = 0
        val_loss = np.Inf #initial value

        with tqdm.tqdm(total=self.num_epochs) as pbar_train:

            for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
                
                step_count+=1
                train_loss_cum = 0 
                train_predictions = []
                train_targets = []
                n_batches = len(self.train_data)

                for idx, (x, y,ids) in enumerate(self.train_data):
                    loss, pred, targets = self.run_train_iter(x=x, y=y)
                    train_loss_cum += loss
                    train_predictions += pred
                    train_targets += targets                
                
                train_loss = train_loss_cum/n_batches
                accuracy = accuracy_score(train_targets, train_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(train_targets, train_predictions, average = 'macro', zero_division=0)
                
                train_stats['train_acc'].append(accuracy)
                train_stats['train_loss'].append(train_loss)
                train_stats['train_f1'].append(f1)
                train_stats['train_recall'].append(recall)
                train_stats['train_precision'].append(precision)
                train_stats['epoch'].append(epoch_idx)
            
                if step_count == self.validation_step:
                    step_count = 0
                    # with tqdm.tqdm(total=len(self.val_data)) as pbar_val:

                    val_loss_cum = 0 
                    val_predictions = []
                    val_targets = []
                    n_batches = len(self.val_data)

                    for idx, (x, y, ids) in enumerate(self.val_data):
                        loss, pred, targets = self.run_evaluation_iter(x=x, y=y)
                        val_loss_cum += loss
                        val_predictions += pred
                        val_targets += targets
                        

                    val_loss = val_loss_cum/n_batches
                    accuracy = accuracy_score(val_targets, val_predictions)
                    precision, recall, f1, _ = precision_recall_fscore_support(val_targets, val_predictions, average = 'macro', zero_division=0)
                    
                    val_stats['val_acc'].append(accuracy)
                    val_stats['val_loss'].append(val_loss)
                    val_stats['val_f1'].append(f1)
                    val_stats['val_recall'].append(recall)
                    val_stats['val_precision'].append(precision)
                    val_stats['epoch'].append(epoch_idx)

                #if the f1-score of the current epoch for the validation set is better than previous epochs
                if f1 > self.best_val_model_f1:
                    self.best_val_model_f1 = f1
                    self.best_val_model_idx = epoch_idx
                    self.save_model(model_save_dir=self.experiment_saved_models,
                        model_save_name="epoch", model_idx=epoch_idx,best_validation_model_idx=self.best_val_model_idx)
                    self.break_epoch = epoch_idx
                    strike = 0
                else:
                    strike+=1

                pbar_train.update(1)
                elapsed_time = time.time() - start_time 
                pbar_train.set_description("Tr/Val loss: {:.3f}/{:.3f}, ET {:.2f}s".format(train_loss,val_loss,elapsed_time))

                if strike == self.patience:
                    print("Early stop, model is overfitting")
                    break

        #save statistics at the end only
        self.save_statistics(experiment_log_dir=self.experiment_logs, filename='training_summary.csv',
                stats_dict=train_stats)  # save statistics    
        self.save_statistics(experiment_log_dir=self.experiment_logs, filename='validation_summary.csv',
                stats_dict=val_stats)  # save statistics    

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
            if self.verbose:
                print("")
                print("Starting test phase")
                print("")
            self.run_test_phase(self.test_data, self.break_epoch,self.experiment_logs)
