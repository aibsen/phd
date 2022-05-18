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
from cv_experiment import CVExperiment
from utils import find_best_epoch
import pandas as pd

class SeededExperiment(nn.Module):
    
    def __init__(self, exp_name,exp_params=None,
        seeds=None,train_data=None,test_data=None,verbose=True,n_seeds=10,low=0,high=10e+5,k=5):

        super(SeededExperiment, self).__init__()

        self.experiment_folder = os.path.abspath(exp_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))


        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)

        self.exp_name = exp_name
        self.exp_params = exp_params
        self.verbose = verbose

        self.k=k
        
        if seeds:
            self.seeds = np.array(seeds)
        else :
            self.seeds = np.random.randint(low,high,n_seeds)

        self.train_data = train_data
        if train_data:
            self.train_length = len(train_data)

        self.test_data = test_data
        if test_data:
            self.test_length = len(test_data)

    def save_seed_statistics(self,summary_list):
        for summary_file in summary_list:
            metrics = None
            for seed in self.seeds:
                exp_name = self.experiment_folder+"/seed_"+str(seed)+"/result_outputs"
                summary = pd.read_csv(exp_name+"/"+summary_file)
                if metrics is None:
                    metrics = summary
                else:
                    metrics=pd.concat([metrics,summary])

            metrics_group = metrics.groupby(metrics.index)
            metrics_means = metrics_group.mean()
            metrics_means.to_csv(self.experiment_logs+"/"+summary_file,index=False)

            means_only = metrics_means[list(filter(lambda k : True if k.find('mean') >=0 else False, list(metrics_means.keys())))]
            best_acc = means_only[means_only.mean_val_acc == means_only.max()['mean_val_acc']][['mean_epoch', 'mean_val_acc']]
            best_f1 = means_only[means_only.mean_val_f1 == means_only.max()['mean_val_f1']][['mean_epoch', 'mean_val_f1']]
            best_loss = means_only[means_only.mean_val_loss == means_only.min()['mean_val_loss']][['mean_epoch', 'mean_val_loss']]
            best_dict = {
                'best_val_acc': best_acc.iloc[0]['mean_val_acc'],
                'best_val_acc_epoch':best_acc.iloc[0]['mean_epoch'],
                'best_val_f1':best_f1.iloc[0]['mean_val_f1'],
                'best_val_f1_epoch':best_f1.iloc[0]['mean_epoch'],
                'best_val_loss':best_loss.iloc[0]['mean_val_loss'],
                'best_val_loss_epoch':best_loss.iloc[0]['mean_epoch'],
            }
            best_values = pd.DataFrame(best_dict,index=[0])
            best_values.to_csv(self.experiment_logs+"/best_"+summary_file,index=False)


    def run_experiment(self, test_results="test_results.csv", test_summary="test_summary.csv"):
        start_time = time.time()

        for seed in self.seeds:
            # print(seed)
            torch.manual_seed(seed=seed)
            print("Starting experiment, seed: "+str(seed))
            exp_name = self.experiment_folder+"/seed_"+str(seed)
            print(exp_name)
            experiment = CVExperiment(exp_name, 
                self.exp_params, 
                self.train_data,
                self.test_data,
                self.verbose,
                k=self.k,
                seed=seed)
            experiment.run_experiment()

        if self.train_data and self.test_data:
            self.save_seed_statistics(["validation_summary.csv",test_summary])
        elif self.train_data:
            self.save_seed_statistics(["validation_summary.csv"])
        else:
            self.save_seed_statistics([test_summary])

        if self.verbose:
            print("--- %s seconds ---" % (time.time() - start_time))
    
    def get_seeds_from_folders(self):
        rootdir = self.experiment_folder
        subdirs = os.walk(rootdir).__next__()[1]
        subdirs = filter(lambda s: True if 'seed' in s else False, subdirs) 
        return list(map(lambda s: s.split("_")[1],subdirs))

    def get_best_results(self, results_filename="test_results.csv", summary_filename="test_summary.csv"):
    #returns the classification results for best iteration
        seeds = self.get_seeds_from_folders()     
        best_seed = -1
        best_k = -1
        best_results = -1
        for seed in seeds:
            exp_name = self.experiment_folder+"/seed_"+str(seed)
            cve = CVExperiment(exp_name)
            k, new_best_results = cve.get_best_fold(summary_filename)
            if new_best_results > best_results:
                best_k = k
                new_best_results = best_results
                best_seed = seed
        r_file = self.experiment_folder+"/seed_"+str(best_seed)+"/folds/fold_k"+str(best_k)+"/result_outputs/"+results_filename
        # results = pd.read_csv(r_file)
        return seed, k, r_file

    def get_all_metrics(self,metric="f1",summary_filename="test_summary.csv"):
    #given a metric it goes through all exp folders and gets the relevant metric
        seeds = self.get_seeds_from_folders()
        summaries = None
        for seed in seeds:
            exp_name = self.experiment_folder+"/seed_"+str(seed)
            cve = CVExperiment(exp_name)
            folds = cve.get_folds_from_folders()
            for fold in folds:
                summary = cve.experiment_folds+"/"+fold+"/result_outputs/"+summary_filename
                s = pd.read_csv(summary)
                if summaries is None:
                    summaries = s
                else :
                    summaries = pd.concat([summaries,s])
        return summaries[metric].values                
