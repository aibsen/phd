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

    def save_seed_statistics(self,summary_file):
        metrics = None
        validation = summary_file == 'validation_summary.csv'
        stats = {'epoch':[],'accuracy':[],'loss':[],'f1':[],'precision':[],'recall':[]}

        if validation:
            stats = {'mean_'+k: v for k, v in stats.items()}

        for seed in self.seeds:
            exp_name = self.experiment_folder+"/seed_"+str(seed)+"/result_outputs"
            summary = pd.read_csv(exp_name+"/"+summary_file)
            for k in stats.keys():
                stats[k].append(summary.iloc[0][k])

        if validation:
            mean_stats = {k: sum(v)/len(v) for k, v in stats.items()}
            stds = {'std_'+k.split('_')[1]: np.std(v) for k, v in stats.items()}
        else:
            mean_stats = {'mean_'+k: sum(v)/len(v) for k, v in stats.items()}
            stds = {'std_'+k: np.std(v) for k, v in stats.items()}

        stats_df = pd.concat([pd.DataFrame(mean_stats, index=[0]), pd.DataFrame(stds, index=[0])], axis=1)
        if not validation:
            keys = [ k for k in stats_df.keys() if 'epoch' not in k]
            stats_df = stats_df[keys]
        
        stats_df.to_csv(self.experiment_logs+"/"+summary_file, index=False)

    def run_experiment(self):
        start_time = time.time()

        for seed in self.seeds:
            torch.cuda.manual_seed(seed=seed)
            print("Starting experiment, seed: "+str(seed))
            exp_name = self.experiment_folder+"/seed_"+str(seed)
            print(exp_name)
            experiment = CVExperiment(exp_name, 
                self.exp_params, 
                self.train_data,
                self.test_data,
                k=self.k,
                seed=seed)
            experiment.run_experiment()

        if self.train_data:
            self.save_seed_statistics("validation_summary.csv")

        if self.test_data:
            self.save_seed_statistics("test_summary.csv")
            self.save_seed_statistics("final_training_summary.csv")

        if self.verbose:
            print("--- %s seconds ---" % (time.time() - start_time))

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
        experiment.run_final_train_phase(data_loaders=[train_loader], n_epochs=self.exp_params["num_epochs"])
        print("--- %s seconds ---" % (time.time() - start_time))

    # def run_test_phase(self):
        
    #     experiment = Experiment(
    #         network_model = self.exp_params["network_model"],
    #         experiment_name = self.experiment_folder,
    #         num_epochs = self.exp_params["num_epochs"],
    #         learning_rate = self.exp_params["learning_rate"],
    #         weight_decay_coefficient = self.exp_params["weight_decay_coefficient"],
    #         batch_size = self.exp_params["batch_size"],
    #         num_output_classes=self.exp_params["num_output_classes"],
    #     )
        
    #     test_loader = torch.utils.data.DataLoader(self.tes6_data, batch_size=self.batch_size, shuffle=False)
        
    #     if self.chunked == True:
    #         experiment.load_model()
    #         experiment.run_test_phase(test_loader, load_model=False)

    
    def get_seeds_from_folders(self):
        rootdir = self.experiment_folder
        subdirs = os.walk(rootdir).__next__()[1]
        subdirs = filter(lambda s: True if 'seed' in s else False, subdirs) 
        return list(map(lambda s: s.split("_")[1],subdirs))

    def save_best_fold_results(self):
        seeds = self.seeds if self.seeds else self.get_seeds_from_folders()
        for seed in seeds:
            exp_name = self.experiment_folder+"/seed_"+str(seed)
            cve = CVExperiment(exp_name, k = self.k)
            cve.save_best_fold_results_f1()

    def save_best_fold_results_acc(self):
        seeds = self.seeds if self.seeds else self.get_seeds_from_folders()
        for seed in seeds:
            exp_name = self.experiment_folder+"/seed_"+str(seed)
            cve = CVExperiment(exp_name, k = self.k)
            cve.save_best_fold_results_acc()
    # def get_best_validation_results(self):
    # #returns the classification results for best iteration
    #     seeds = self.get_seeds_from_folders()     
    #     best_seed = -1
    #     best_k = -1
    #     best_results = -1
    #     for seed in seeds:
    #         exp_name = self.experiment_folder+"/seed_"+str(seed)
    #         cve = CVExperiment(exp_name)
    #         k, new_best_results = cve.get_best_fold(summary_filename)
    #         if new_best_results > best_results:
    #             best_k = k
    #             new_best_results = best_results
    #             best_seed = seed
    #     r_file = self.experiment_folder+"/seed_"+str(best_seed)+"/folds/fold_k"+str(best_k)+"/result_outputs/"+results_filename
    #     # results = pd.read_csv(r_file)
    #     return seed, k, r_file

    # def get_all_metrics(self,metric="f1",summary_filename="test_summary.csv"):
    # #given a metric it goes through all exp folders and gets the relevant metric
    #     seeds = self.get_seeds_from_folders()
    #     summaries = None
    #     for seed in seeds:
    #         exp_name = self.experiment_folder+"/seed_"+str(seed)
    #         cve = CVExperiment(exp_name)
    #         folds = cve.get_folds_from_folders()
    #         for fold in folds:
    #             summary = cve.experiment_folds+"/"+fold+"/result_outputs/"+summary_filename
    #             s = pd.read_csv(summary)
    #             if summaries is None:
    #                 summaries = s
    #             else :
    #                 summaries = pd.concat([summaries,s])
    #     return summaries[metric].values                
