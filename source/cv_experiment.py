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

class CVExperiment(nn.Module):
    def __init__(self, exp_name,exp_params,train_data=None,test_data=None,k=5,verbose=True):

        super(CVExperiment, self).__init__()
        self.experiment_folder = os.path.abspath(exp_name)
        print(self.experiment_folder)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        print(self.experiment_logs)
        self.experiment_folds = os.path.abspath(os.path.join(self.experiment_folder, "folds"))
        print(self.experiment_folds)


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
            self.test_loader = torch.utils.data.DataLoader(self.test_data,batch_size=self.exp_params["batch_size"],shuffle=True)
        else:
            self.test_loader=None


    def save_fold_statistics(self,summary_list):
        for summary_file in summary_list:
            metrics = []
            for k in np.arange(self.k):
                exp_name = self.experiment_folds+"/fold_k"+str(k+1)+"/result_outputs"
                summary = load_statistics(exp_name,summary_file)
                metrics.append(summary)

            keys = metrics[0].keys()
            stats_keys = [[k+"_mean",k+"_std"] for k in keys]
            stats_keys = [item for sublist in stats_keys for item in sublist]
            stats = {key: [] for key in stats_keys}

            for k in keys:
                metric = [m[k] for m in metrics]
                meank = k+"_mean"
                stdk = k+"_std"
                stats[meank] = [np.array(metric).astype('float64').mean()]
                stats[stdk] = [np.array(metric).astype('float64').std()]
            save_statistics(experiment_log_dir=self.experiment_logs, filename=summary_file,stats_dict=stats, current_epoch=0)


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
            print(len(train_dataset))
            print(len(val_dataset))
            print("")

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.exp_params["batch_size"], shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.exp_params["batch_size"], shuffle=True)

            experiment = Experiment(
                network_model = self.exp_params["network_model"],
                experiment_name = self.experiment_folds+"/fold_k"+str(k+1),
                num_epochs = self.exp_params["num_epochs"],
                learning_rate = self.exp_params["learning_rate"],
                weight_decay_coefficient = self.exp_params["weight_decay_coefficient"],
                use_gpu = self.exp_params["use_gpu"],
                train_data = train_loader,
                val_data = val_loader,
                test_data = self.test_loader,
                verbose = self.verbose
            )

            start_time = time.time()
            experiment.run_experiment(test_results=test_results,test_summary=test_summary)
            print("--- %s seconds ---" % (time.time() - start_time))

    def run_test_phase(self,test_results="test_results.csv",test_summary="test_summary.csv"):
        for k in np.arange(self.k):
            exp_name = self.experiment_folds+"/fold_k"+str(k+1),
            best_epoch = find_best_epoch(exp_name+"/result_outputs/summary.csv")

            experiment = Experiment(
                network_model = self.params["network_model"],
                experiment_name = exp_name,
                use_gpu = self.params["use_gpu"],
                test_data = test_loader,
                best_idx = best_epoch,
                verbose = self.verbose
            )
            start_time = time.time()
            experiment.run_test_phase(self.test_loader,test_results,test_summary)
            print("--- %s seconds ---" % (time.time() - start_time))



    