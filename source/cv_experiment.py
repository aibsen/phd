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
from datasets import LCs, CachedLCs
from data_samplers import CachedRandomSampler
from dataset_utils import cached_crossvalidator_split
from experiment import Experiment
from utils import find_best_epoch
from plot_utils import *
from sklearn.metrics import f1_score, accuracy_score

import pandas as pd

class CVExperiment(nn.Module):
    def __init__(self, exp_name,exp_params=None,train_data=None,test_data=None,k=5,seed=None):

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

        self.exp_params = exp_params
        self.k = k
        self.train_data = train_data
        self.test_data = test_data
        self.best_val_fold = None
        self.best_fold = None
        # self.batch_size = self.exp_params['batch_size']
        self.mean_best_epoch = None
        
        if train_data:
            self.train_length = len(train_data)
            idxs = np.arange(self.train_length)
            targets = train_data.targets
            kf = StratifiedKFold(n_splits=k)
            self.kfs = kf.split(idxs,targets)

    def save_fold_statistics(self, summary_file, metric='f1'):
        stats = {'epoch':[],'accuracy':[],'loss':[],'f1':[],'precision':[],'recall':[]}
        for k in np.arange(self.k):
            exp_name = self.experiment_folds+"/fold_k"+str(k+1)+"/result_outputs"
            summary = pd.read_csv(exp_name+"/"+summary_file)
            stats_at_break = summary[summary[metric] == summary[metric].max()]
            for k in stats.keys():
                stats[k].append(stats_at_break.iloc[0][k])

        best_fold = stats[metric].index(max(stats[metric]))+1    
        stats = {k: sum(v)/len(v) for k, v in stats.items()}
        stats_df =  pd.DataFrame(stats, index=[0]).rename(columns = lambda c : 'mean_'+ c)
        stats_df.mean_epoch = stats_df.mean_epoch.apply(np.ceil).astype(int)
        if summary_file == 'validation_summary.csv':
            self.best_fold = best_fold    
            self.mean_best_epoch = int(stats_df.iloc[0]['mean_epoch'])
        stats_df.to_csv(self.experiment_logs+"/"+summary_file,index=False)

    def run_experiment(self, test_data_name="test"):
        if self.train_data:
            self.run_train_phase()
        if self.train_data and self.test_data:
            self.run_final_train_phase()
            self.run_test_phase(test_data_name)
        elif self.test_data:
            self.run_test_phase(test_data_name)

    def run_train_phase(self):
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
                validation_step = self.exp_params["validation_step"]
            )

            # start_time = time.time()
            experiment.run_train_phase()
            # print("--- %s seconds ---" % (time.time() - start_time))


        self.save_fold_statistics("validation_summary.csv")
        self.save_fold_statistics("training_summary.csv") #there's an error here that I don't have time to fix: I'm saving best training stats, but kept last training epoch results.

        #final run of cv experiment??
    def run_final_train_phase(self):

        experiment = Experiment(
            network_model = self.exp_params["network_model"],
            experiment_name = self.experiment_folder,
            num_epochs = self.exp_params["num_epochs"],
            learning_rate = self.exp_params["learning_rate"],
            weight_decay_coefficient = self.exp_params["weight_decay_coefficient"],
            batch_size = self.exp_params["batch_size"],
            num_output_classes=self.exp_params["num_output_classes"],
        )

        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        start_time = time.time()
        # print(self.mean_best_epoch)
        if not self.mean_best_epoch:
            self.mean_best_epoch = self.get_mean_best_epoch()

        experiment.run_final_train_phase(data_loaders=[train_loader], n_epochs=self.mean_best_epoch)
        # print("--- %s seconds ---" % (time.time() - start_time))

    def run_test_phase(self, test_data_name='test'):

        exp_name = self.experiment_folder
        experiment = Experiment(
            network_model = self.exp_params["network_model"],
            experiment_name = exp_name,
            batch_size = self.exp_params["batch_size"],
            num_output_classes= self.exp_params["num_output_classes"],
        )

        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        # start_time = time.time()
        experiment.run_test_phase(data=test_loader,data_name=test_data_name)
        # print("--- %s seconds ---" % (time.time() - start_time))


    def get_mean_best_epoch(self):
        val_summary = pd.read_csv(self.experiment_logs+"/"+"validation_summary.csv")
        best_mean_epoch = val_summary.iloc[0]['mean_epoch']
        return best_mean_epoch

    def get_fold_results(self, fold):
        fold_dir = self.experiment_folds+"/fold_k{}/result_outputs/".format(fold)
        results_path = fold_dir+"validation_results.csv"
        results = pd.read_csv(results_path)
        predictions = results.prediction
        targets = results.target
        return predictions.values,targets.values

    def save_best_fold_results_acc(self, plot=True):
        cum_acc_scores = []

        for fold in range(1,self.k+1):
            predictions, targets = self.get_fold_results(fold)
            cum_acc_scores.append(accuracy_score(targets,predictions))

        best_acc = {}
        best_fold_idx = cum_acc_scores.index(np.max(cum_acc_scores))
        best_acc['fold']=best_fold_idx+1
        best_acc['accuracy']=cum_acc_scores[best_fold_idx]
            
        best_acc_scores = pd.DataFrame(best_acc, index = [0])
        best_acc_scores.to_csv(self.experiment_logs+"/"+"best_acc_scores.csv",index=False)

    def save_best_fold_results_f1(self, plot=True):
        reduction = ["micro","macro","weighted"]
        cum_f1_scores = {r:[] for r in reduction}

        for fold in range(1,self.k+1):
            predictions, targets = self.get_fold_results(fold)
            fold_dir = self.experiment_folds+"/fold_k{}/result_outputs/".format(fold)
            f1_scores = {r:f1_score(targets,predictions,average=r) for r in reduction}
            f1_scores_df = pd.DataFrame(f1_scores, index=[0])
            f1_scores_df.to_csv(fold_dir+"f1_scores.csv",index=False)
            for r in reduction:
                cum_f1_scores[r].append(f1_scores[r])
        
        avg_f1_scores = {r:np.mean(cum_f1_scores[r]) for r in cum_f1_scores}
        avg_f1_scores_df = pd.DataFrame(avg_f1_scores, index =[0])
        avg_f1_scores_df.to_csv(self.experiment_logs+"/"+"mean_f1_scores.csv",index=False)

        best_f1_per_reduction = {r:[] for r in ['reduction','fold']+reduction}
        for r in reduction:
            best_fold_idx = cum_f1_scores[r].index(np.max(cum_f1_scores[r]))
            best_f1_per_reduction['reduction'].append(r)
            best_f1_per_reduction['fold'].append(best_fold_idx+1)
            for red in reduction:
                best_f1_per_reduction[red].append(cum_f1_scores[red][best_fold_idx])
            
            if plot: 
                self.save_result_plots(self.experiment_logs,fold=best_fold_idx+1,plot_name="best_cm_{}_fold_{}.png".format(r,best_fold_idx+1))
                print(r) 
        best_f1_scores = pd.DataFrame(best_f1_per_reduction)
        best_f1_scores.to_csv(self.experiment_logs+"/"+"best_f1_scores.csv")
                     
    def save_result_plots(self,save_dir,fold,plot_name="best_cm.png"):
        save_path = save_dir+"/"+plot_name
        predictions, targets = self.get_fold_results(fold)
        plot_best_val_cm(targets,predictions, save=True,verbose=False, output_file=save_path)

    # def get_folds_from_folders(self):
    #     rootdir = self.experiment_folds
    #     folds = os.walk(rootdir).__next__()[1]
    #     return folds


    # def get_best_fold(self,summary_filename="test_summary.csv",metric="f1"):
    #     folds = self.get_folds_from_folders()
    #     best_k = -1
    #     best_metric = -1
    #     for fold in folds:
    #         summary = self.experiment_folds+"/"+fold+"/result_outputs/"+summary_filename
    #         summary_df=pd.read_csv(summary)
    #         new_metric = summary_df[metric].values
    #         # print(summary_df)
    #         # print(new_metric)
    #         if new_metric>best_metric: #this would only work for acc and f1
    #             best_metric = new_metric
    #             best_k = fold[-1] #only works for 9 or less folds
        
    #     return best_k, best_metric

    