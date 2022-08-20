from random import randint
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

        self.seeds = np.array(seeds) if seeds else np.random.randint(low, high, n_seeds)

        name_template = self.experiment_folder+'/seed_' 
        self.cv_experiments = [CVExperiment(name_template+str(seed), exp_params, train_data, test_data, k=k, seed=seed) for seed in self.seeds]
        self.train_data = train_data
        self.test_data = test_data



    def save_seed_statistics(self,summary_file):
        # validation = summary_file == 'validation_summary.csv'
        stats = {'epoch':[],'accuracy':[],'loss':[],'f1':[],'precision':[],'recall':[]}

        # if validation:
        stats = {'mean_'+k: v for k, v in stats.items()}

        for seed in self.seeds:
            exp_name = self.experiment_folder+"/seed_"+str(seed)+"/result_outputs"
            summary = pd.read_csv(exp_name+"/"+summary_file)
            for k in stats.keys():
                stats[k].append(summary.iloc[-1][k])

        # if validation:
        mean_stats = {k: sum(v)/len(v) for k, v in stats.items()}
        stds = {'std_'+k.split('_')[1]: np.std(v) for k, v in stats.items()}
        # else:
            # mean_stats = {'mean_'+k: sum(v)/len(v) for k, v in stats.items()}
            # stds = {'std_'+k: np.std(v) for k, v in stats.items()}

        stats_df = pd.concat([pd.DataFrame(mean_stats, index=[0]), pd.DataFrame(stds, index=[0])], axis=1)
        # if not validation:
            # keys = [ k for k in stats_df.keys() if 'epoch' not in k]
            # stats_df = stats_df[keys]
        
        stats_df.to_csv(self.experiment_logs+"/"+summary_file, index=False)

    def run_experiment(self):
        start_time = time.time()
        for i, seed in enumerate(self.seeds):
            torch.manual_seed(seed=seed)
            print("Starting experiment, seed: "+str(seed))
            self.cv_experiments[i].run_experiment()

        if self.train_data:
            self.save_seed_statistics("validation_summary.csv")

        if self.test_data:
            self.save_seed_statistics("test_summary.csv")

        if self.verbose:
            print("--- %s seconds ---" % (time.time() - start_time))

    def run_test_phase(self, save_name = 'test'):
        if len(self.cv_experiments) > 0:
            for i, seed in enumerate(self.seeds):
                torch.manual_seed(seed=seed)
                print("Starting test phase, seed: "+str(seed))
                self.cv_experiments[i].test_data = self.test_data
                self.cv_experiments[i].run_test_phase(save_name=save_name)
                del self.cv_experiments[i].test_data

            self.save_seed_statistics("test{}_summary.csv".format(save_name))
    
    def get_seeds_from_folders(self):
        rootdir = self.experiment_folder
        subdirs = os.walk(rootdir).__next__()[1]
        subdirs = filter(lambda s: True if 'seed' in s else False, subdirs) 
        return list(map(lambda s: s.split("_")[1],subdirs))
