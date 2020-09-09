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
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support
from utils import save_to_stats_pkl_file, load_from_stats_pkl_file, \
    save_statistics, load_statistics, save_classification_results


class Experiment(nn.Module):
    def __init__(self, network_model, 
        experiment_name,metric="f1_score", 
        num_epochs=100, 
        learning_rate=1e-03,
        batch_size = 64, 
        train_data=None, 
        val_data=None,
        test_data=None,
        balance_training_set=False,
        weight_decay_coefficient=0, 
        use_gpu=True, 
        continue_from_epoch=-1, 
        num_output_classes=4, 
        best_idx=0, 
        verbose=True,
        cached_dataset=False):

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
        self.num_output_classes = num_output_classes


        if train_data and val_data:
            if balance_training_set:
                weights, num_samples = self.calculate_balance_weights(train_data)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples)                     
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler)     
            else:
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
            self.train_data = train_loader
            self.val_data = val_loader
        else:
            self.train_data = None
            self.val_data = None

        if test_data:
            test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)
            self.test_data = test_loader

        else:
            self.test_data = None

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=False,
                                    weight_decay=weight_decay_coefficient)

        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = best_idx
        self.best_val_model_acc = 0
        self.best_val_model_f1 = 0

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU

        if continue_from_epoch != -1:  # if continue from epoch is not -1 then
            try:
                self.best_val_model_idx, self.best_val_model_acc, self.best_val_model_f1 = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model_"+self.metric,
                    model_idx=continue_from_epoch)  # reload existing model from epoch
                self.starting_epoch = continue_from_epoch
            except:
                print("Did not save that epoch's model, will start from 0")
                self.starting_epoch = 0
        else:
            self.starting_epoch = 0

    def calculate_balance_weights(self, train_data):
        counts = torch.zeros(self.num_output_classes)
        labels = torch.zeros(len(train_data), dtype=torch.long)
        for i,item in enumerate(train_data):
            counts[item[1]] +=1 
            labels[i] = item[1]

        weights_per_class = 1/counts
        weights = weights_per_class[labels]
        num_samples = int(counts.min()*self.num_output_classes)
        return weights, num_samples


    def run_train_iter(self, x, y):
        self.train()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        out = self.model.forward(x)  # forward the data in the model
        loss = self.criterion(out,y)
        loss.backward()  # backpropagate
        self.optimizer.step()
        predicted = torch.argmax(out.data, 1)
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        loss = loss.data.cpu()
        y_cpu = y.data.cpu()
        predicted_cpu = predicted.cpu()
        p,r,f1_score,s = precision_recall_fscore_support(y_cpu,predicted_cpu, average='weighted', labels=np.unique(predicted_cpu))
        return loss,accuracy,f1_score,p,r

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode
        out = self.model.forward(x)  # forward the data in the model
        loss =  self.criterion(out,y)
        predicted = torch.argmax(out.data, 1)
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        loss = loss.data.cpu()
        y_cpu = y.data.cpu()
        predicted_cpu = predicted.cpu()
        p,r,f1_score,s = precision_recall_fscore_support(y_cpu,predicted_cpu, average='weighted', labels=np.unique(predicted_cpu))
        return loss, accuracy,f1_score,p,r,out.data

    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx):
                #    best_validation_model_acc, best_validation_model_f1):
        state = dict()
        state['network'] = self.model.state_dict()  # save network parameter and other variables.
        state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        # state['best_val_model_acc'] = best_validation_model_acc  # save current best val acc
        # state['best_val_model_f1'] = best_validation_model_f1  # save current best val acc
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))

    def load_model(self, model_save_dir, model_save_name, model_idx):
        filename = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
        if self.verbose:
            print(filename)
        state = torch.load(f=filename)
        self.model.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx']#, state['best_val_model_acc'], state['best_val_model_f1']

    def run_test_phase(self, data, results_filename, summary_filename):
        #getting evaluation metrics for best epoch model only
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,model_save_name="train_model_"+self.metric)
        metrics = {"acc": [], "loss": [], "f1": [],"precision":[],"recall":[]}
        soft_results = None
        actual_tags = None
        id_events = None

        # if self.verbose:
        #     pbar = tqdm.tqdm(total=len(data))
        with tqdm.tqdm(total=len(data)) as pbar:
            for x, y, ids in data:
                loss, accuracy,f1,p,r,results = self.run_evaluation_iter(x=x,y=y)
                metrics["loss"].append(loss)
                metrics["acc"].append(accuracy)
                metrics["f1"].append(f1)
                metrics["precision"].append(p)
                metrics['recall'].append(r)
                results = F.softmax(results,dim=1)

                if soft_results is None:
                    soft_results = results
                    actual_tags = y
                    id_events =ids
                else :
                    soft_results = torch.cat((soft_results, results),0)
                    actual_tags = torch.cat((actual_tags, y),0)
                    id_events = torch.cat((id_events,ids),0)
                # if self.verbose:
                pbar.update(1)

        total_metrics = {key: [np.mean(value)] for key, value in metrics.items()}  # save vaidation set metrics
        if self.verbose:
            [print("    ",key,": ",str(value)) for key, value in total_metrics.items()]

        save_classification_results(experiment_log_dir=self.experiment_logs, filename=results_filename,results=soft_results,
            ids=id_events,tags=actual_tags,n_classes=self.num_output_classes)
        save_statistics(experiment_log_dir=self.experiment_logs, filename=summary_filename,stats_dict=total_metrics, current_epoch=0)

    def run_train_phase(self):
        total_losses = {"train_acc": [], "train_loss": [], "train_f1":[],"train_precision":[],"train_recall":[], "val_acc": [],
                        "val_loss": [], "val_f1":[], "val_precision":[],"val_recall":[]}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_metrics = {"train_acc": [], "train_loss": [], "train_f1":[], "train_precision":[],"train_recall":[],
            "val_acc": [], "val_loss": [],"val_f1":[],"val_precision":[],"val_recall":[]}

            # if self.verbose:
            #     pbar_train = tqdm.tqdm(total=len(self.train_data))
            #     pbar_val = tqdm.tqdm(total=len(self.val_data))
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:
                for idx, (x, y,ids) in enumerate(self.train_data):
                    loss, accuracy,f1, p, r = self.run_train_iter(x=x, y=y)
                    current_epoch_metrics["train_loss"].append(loss)
                    current_epoch_metrics["train_acc"].append(accuracy)
                    current_epoch_metrics["train_f1"].append(f1)
                    current_epoch_metrics["train_precision"].append(p)
                    current_epoch_metrics['train_recall'].append(r)
                    # if self.verbose:
                    pbar_train.update(1)
                # pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}, f1_score: {:.4f}".format(loss, accuracy, f1))

            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:
                for x, y,ids in self.val_data:
                    loss, accuracy,f1,p,r,_ = self.run_evaluation_iter(x=x, y=y)
                    current_epoch_metrics["val_loss"].append(loss)
                    current_epoch_metrics["val_acc"].append(accuracy)
                    current_epoch_metrics["val_f1"].append(f1)
                    current_epoch_metrics["val_precision"].append(p)
                    current_epoch_metrics['val_recall'].append(r)
                    # if self.verbose:
                    pbar_val.update(1)
                # pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}, f1_score: {:.4f}".format(loss, accuracy,f1))

            if self.metric == "accuracy":
                val_mean_accuracy = np.mean(current_epoch_metrics['val_acc'])
                # if val_mean_accuracy > self.best_val_model_acc:
                if val_mean_accuracy > self.best_val_model_idx:
                    # self.best_val_model_acc = val_mean_accuracy
                    self.best_val_model_idx = epoch_idx

            elif self.metric == "f1_score":
                val_mean_f1 = np.mean(current_epoch_metrics['val_f1'])
                # if val_mean_f1 > self.best_val_model_f1:
                if val_mean_f1 > self.best_val_model_idx:
                    # self.best_val_model_f1 = val_mean_f1
                    self.best_val_model_idx = epoch_idx

            self.save_model(model_save_dir=self.experiment_saved_models,
                model_save_name="train_model_"+self.metric, model_idx=epoch_idx,
                best_validation_model_idx=self.best_val_model_idx,
                # best_validation_model_acc=self.best_val_model_acc,
                # best_validation_model_f1=self.best_val_model_f1
                )

            for key, value in current_epoch_metrics.items():

                total_losses[key].append(np.mean(value))  # get mean of all metrics of current epoch
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i)  # save statistics

            if self.verbose:
                out_string = "\n    ".join(["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_metrics.items()])

                epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
                epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
                print("Epoch {}:\n".format(epoch_idx), out_string, "\n epoch time", epoch_elapsed_time, "seconds")

    def run_experiment(self,test_results="test_results.csv", test_summary="test_summary.csv"):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        if self.train_data and self.val_data:
            self.run_train_phase()
            if self.verbose:
                print("Starting training phase")
        #getting evaluation metrics for best epoch model only
            self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx, model_save_name="train_model_"+self.metric)
            if self.verbose:
                print("Starting test phase")
                print("Generating val set evaluation metrics")
            self.run_test_phase(self.val_data, "validation_results.csv", "validation_summary.csv")

        if self.test_data:
            if self.verbose:
                print("Generating test set evaluation metrics")
            self.run_test_phase(self.test_data, test_results, test_summary)
