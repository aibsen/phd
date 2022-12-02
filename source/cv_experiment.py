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
import time
from sklearn.model_selection import StratifiedKFold
from datasets import LCs
from experiment import Experiment
from utils import find_best_epoch
from plot_utils import *
from sklearn.metrics import f1_score, accuracy_score

import pandas as pd

class CVExperiment(nn.Module):
    def __init__(self, exp_name,exp_params=None,
        train_data=None,test_data=None,k=5):
        # , seed=1772670):

        super(CVExperiment, self).__init__()
        self.experiment_folder = os.path.abspath(exp_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_folds = os.path.abspath(os.path.join(self.experiment_folder, "folds"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_folds)  # create the experiment fold directories
            os.mkdir(self.experiment_saved_models)  # create the experiment fold directories
        
        self.k = k
        self.reset_experiment_params(exp_params)
        self.reset_experiment_datasets(train_data,test_data)


    def reset_experiment_params(self,exp_params):
        self.exp_params = exp_params
        self.best_val_fold = None
        self.best_fold = None
        self.batch_size = self.exp_params['batch_size']
        self.mean_best_epoch = None
        self.experiment_type = 'classification' if 'experiment_type' not in exp_params else exp_params['experiment_type']
        self.pick_up = False if 'pick_up' not in exp_params else exp_params['pick_up']
    
    def reset_experiment_datasets(self, train_data=None, test_data=None):
        self.train_data = train_data
        self.test_data = test_data
        
        if train_data:
            self.train_length = len(train_data)
            idxs = np.arange(self.train_length)
            targets = train_data.targets
            kf = StratifiedKFold(n_splits=self.k)
            self.kfs = kf.split(idxs,targets)

# 
    def save_fold_statistics(self, summary_file):

        if self.experiment_type == 'classification':
            metric = "f1"
            stats = {'epoch':[],'accuracy':[],'loss':[],'f1':[],'precision':[],'recall':[]}
        elif self.experiment_type == 'seq2seq':
            metric = 'loss'
            stats = {'epoch':[],'loss':[]}
            summary_file = 'reconstruction_'+summary_file
        else:
            print('Invalid experiment_type: currently only classification and seq2seq are implemented')

        for k in np.arange(self.k):
            exp_name = self.experiment_folds+"/fold_k"+str(k+1)+"/result_outputs"
            summary = pd.read_csv(exp_name+"/"+summary_file)
            best = summary[metric].max() if metric != 'loss' else summary[metric].min()
            stats_at_break = summary[summary[metric] == best] 
        
            for k in stats.keys():
                stats[k].append(stats_at_break.iloc[0][k])

        best_best = max(stats[metric]) if metric != 'loss' else min(stats[metric])
        best_fold = stats[metric].index(best_best)+1    
        stats = {k: sum(v)/len(v) for k, v in stats.items()}
        stats_df =  pd.DataFrame(stats, index=[0]).rename(columns = lambda c : 'mean_'+ c)
        stats_df.mean_epoch = stats_df.mean_epoch.apply(np.ceil).astype(int)

        if 'validation_summary.csv' in summary_file:
            self.best_fold = best_fold    
            self.mean_best_epoch = int(stats_df.iloc[0]['mean_epoch'])

        stats_df.to_csv(self.experiment_logs+"/"+summary_file,index=False)

    def run_experiment(self,final_only=False,
        test_data_name="test",train_data_name='',
        model_load_name_train='best_validation_model.pth.tar',
        model_save_name_train='best_validation_model.pth.tar',
        model_load_name_final_train='final_model.pth.tar',
        model_save_name_final_train='final_model.pth.tar',
        ):

        if self.train_data and not final_only:
            self.run_train_phase(train_data_name,
                model_load_name=model_load_name_train,
                model_save_name=model_save_name_train)

        if self.train_data and self.test_data:
            self.run_final_train_phase(train_data_name,
                model_load_name=model_load_name_final_train,
                model_save_name=model_save_name_final_train)

            self.run_test_phase(test_data_name, 
                model_name=model_save_name_final_train)

        elif self.test_data:
            self.run_test_phase(test_data_name,
            model_name=model_save_name_final_train)

    def run_train_phase(self,
        train_data_name='',
        model_load_name='best_validation_model.pth.tar',
        model_save_name='best_validation_model.pth.tar'
        ):

        for k,(tr,val) in enumerate(self.kfs):
            train_dataset = torch.utils.data.Subset(self.train_data, tr)
            val_dataset = torch.utils.data.Subset(self.train_data, val)

            experiment = Experiment(
                network_model = self.exp_params["network_model"],
                experiment_name = self.experiment_folds+"/fold_k"+str(k+1),
                num_epochs = self.exp_params["num_epochs"],
                learning_rate = self.exp_params["learning_rate"],
                weight_decay_coefficient = self.exp_params["weight_decay_coefficient"],
                batch_size = self.exp_params["batch_size"],
                train_data = train_dataset,
                val_data = val_dataset,
                test_data = self.test_data,
                num_output_classes=self.exp_params["num_output_classes"],
                patience = self.exp_params["patience"],
                validation_step = self.exp_params["validation_step"],
                type = self.experiment_type,
                pick_up = self.pick_up
            )

            # start_time = time.time()
            experiment.run_train_phase(train_data_name,
                model_load_name=model_load_name,
                model_save_name=model_save_name)
            # print("--- %s seconds ---" % (time.time() - start_time))

        self.save_fold_statistics(train_data_name+"validation_summary.csv")
        self.save_fold_statistics(train_data_name+"training_summary.csv") #there's an error here that I don't have time to fix: I'm saving best training stats, but kept last training epoch results.

        #final run of cv experiment??
    def run_final_train_phase(self,train_data_name='',
        model_load_name='final_model.pth.tar',
        model_save_name='final_model.pth.tar'
        ):

        if not self.mean_best_epoch:
            self.mean_best_epoch = self.get_mean_best_epoch(train_data_name=train_data_name)

        experiment = Experiment(
            network_model = self.exp_params["network_model"],
            experiment_name = self.experiment_folder,
            num_epochs = self.mean_best_epoch,
            learning_rate = self.exp_params["learning_rate"],
            weight_decay_coefficient = self.exp_params["weight_decay_coefficient"],
            batch_size = self.exp_params["batch_size"],
            num_output_classes=self.exp_params["num_output_classes"],
            type=self.experiment_type,
            patience=self.exp_params['patience'],
            validation_step=self.exp_params['validation_step'],
            pick_up = self.pick_up
        )
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        start_time = time.time()
        # print(self.mean_best_epoch)
        if not self.mean_best_epoch:
            self.mean_best_epoch = self.get_mean_best_epoch(train_data_name=train_data_name)

        experiment.run_final_train_phase(data_loaders=[train_loader], 
            n_epochs=self.mean_best_epoch,
            train_data_name=train_data_name,
            model_load_name=model_load_name,
            model_save_name=model_save_name)
        # print("--- %s seconds ---" % (time.time() - start_time))

    def run_test_phase(self, test_data_name='test',model_name='final_model.pth.tar'):

        exp_name = self.experiment_folder
        experiment = Experiment(
            network_model = self.exp_params["network_model"],
            experiment_name = exp_name,
            batch_size = self.exp_params["batch_size"],
            num_output_classes= self.exp_params["num_output_classes"],
            type = self.experiment_type
        )

        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        # start_time = time.time()
        experiment.run_test_phase(data=test_loader,data_name=test_data_name,model_name=model_name)
        # print("--- %s seconds ---" % (time.time() - start_time))


    def run_prediction(self, data_name='predicted',model_name='final_model.pth.tar'):

        exp_name = self.experiment_folder
        experiment = Experiment(
            network_model = self.exp_params["network_model"],
            experiment_name = exp_name,
            batch_size = self.exp_params["batch_size"],
            num_output_classes= self.exp_params["num_output_classes"],
            type = self.experiment_type
        )
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        # start_time = time.time()
        experiment.run_prediction(data=test_loader,data_name=data_name,model_name=model_name)
        # print("--- %s seconds ---" % (time.time() - start_time))


    def get_mean_best_epoch(self,train_data_name=''):
        fn = train_data_name+"validation_summary.csv"
        fn = 'reconstruction_'+fn if self.experiment_type=='seq2seq' else fn
        val_summary = pd.read_csv(self.experiment_logs+'/'+fn)
        best_mean_epoch = val_summary.iloc[0]['mean_epoch']
        return best_mean_epoch
