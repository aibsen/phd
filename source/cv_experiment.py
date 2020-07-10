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
from sklearn.model_selection import KFold
from datasets import LCs
from experiment import Experiment
from utils import load_statistics,save_statistics,find_best_epoch
import pandas as pd

class CVExperiment(nn.Module):
    def __init__(self, exp_name,exp_params=None,train_data=None,test_data=None,verbose=True,k=5):

        super(CVExperiment, self).__init__()
        self.experiment_folder = os.path.abspath(exp_name)
        # print(self.experiment_folder)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        # print(self.experiment_logs)
        self.experiment_folds = os.path.abspath(os.path.join(self.experiment_folder, "folds"))
        # print(self.experiment_folds)


        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_folds)  # create the experiment saved models directory

        self.exp_params = exp_params
        self.k = k
        self.verbose = verbose
        self.train_data = train_data
        if train_data:
            self.train_length = len(train_data)
            idxs = np.arange(self.train_length)
            kf = KFold(n_splits=k)
            self.kfs = kf.split(idxs)

        self.test_data = test_data
        if test_data:
            self.test_length = len(test_data)
        else:
            self.test_loader=None


    def save_fold_statistics(self,summary_list):
        for summary_file in summary_list:
            metrics = None
            for k in np.arange(self.k):
                exp_name = self.experiment_folds+"/fold_k"+str(k+1)+"/result_outputs"
                summary = pd.read_csv(exp_name+"/"+summary_file)
                if metrics is None:
                    metrics = summary
                else:
                    metrics=pd.concat([metrics,summary])
            metrics_group = metrics.groupby(metrics.index)
            metrics_means = metrics_group.mean().rename(columns = lambda x : 'mean_' + x)
            metrics_stds = metrics_group.std().rename(columns = lambda x : 'std_' + x)
            all_metrics = pd.concat([metrics_means,metrics_stds],axis=1)
            all_metrics.to_csv(self.experiment_logs+"/"+summary_file,index=False)

    def run_experiment(self, test_results="test_results.csv", test_summary="test_summary.csv"):
        if self.train_data:
            self.run_train_phase(test_results,test_summary)
            if self.test_data:
                self.save_fold_statistics(["validation_summary.csv",test_summary])
            else:
                self.save_fold_statistics(["validation_summary.csv"])
        elif self.test_data:
            self.run_test_phase(test_results, test_summary)
            self.save_fold_statistics([test_summary])

    def run_train_phase(self, test_results="test_results.csv", test_summary="test_summary.csv"):
        for k,(tr,val) in enumerate(self.kfs):

            train_dataset = torch.utils.data.Subset(self.train_data, tr)
            val_dataset = torch.utils.data.Subset(self.train_data, val)
         
            experiment = Experiment(
                network_model = self.exp_params["network_model"],
                experiment_name = self.experiment_folds+"/fold_k"+str(k+1),
                num_epochs = self.exp_params["num_epochs"],
                learning_rate = self.exp_params["learning_rate"],
                weight_decay_coefficient = self.exp_params["weight_decay_coefficient"],
                use_gpu = self.exp_params["use_gpu"],
                batch_size = self.exp_params["batch_size"],
                balance_training_set = self.exp_params["balance_training_set"],
                train_data = train_dataset,
                val_data = val_dataset,
                test_data = self.test_data,
                verbose = self.verbose
            )

            start_time = time.time()
            experiment.run_experiment(test_results=test_results,test_summary=test_summary)
            if self.verbose:
                print("--- %s seconds ---" % (time.time() - start_time))

    def run_test_phase(self,test_results="test_results.csv",test_summary="test_summary.csv"):
        for k in np.arange(self.k):
            exp_name = self.experiment_folds+"/fold_k"+str(k+1)
            print(exp_name)
            best_epoch = find_best_epoch(exp_name+"/result_outputs/summary.csv")

            experiment = Experiment(
                network_model = self.exp_params["network_model"],
                experiment_name = exp_name,
                use_gpu = self.exp_params["use_gpu"],
                test_data = self.test_data,
                best_idx = best_epoch,
                verbose = self.verbose
            )
            start_time = time.time()
            experiment.run_experiment(test_results,test_summary)
            if self.verbose:
                print("--- %s seconds ---" % (time.time() - start_time))


    def get_folds_from_folders(self):
        rootdir = self.experiment_folds
        folds = os.walk(rootdir).__next__()[1]
        return folds


    def get_best_fold(self,summary_filename="test_summary.csv",metric="f1"):
        folds = self.get_folds_from_folders()
        best_k = -1
        best_metric = -1
        for fold in folds:
            summary = self.experiment_folds+"/"+fold+"/result_outputs/"+summary_filename
            summary_df=pd.read_csv(summary)
            new_metric = summary_df[metric].values
            # print(summary_df)
            # print(new_metric)
            if new_metric>best_metric: #this would only work for acc and f1
                best_metric = new_metric
                best_k = fold[-1] #only works for 9 or less folds
        
        return best_k, best_metric

    