from torch import nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#import torchvision
import tqdm
import os
import numpy as np
import time
# from mlp.pytorch_experiment_scripts.storage_utils
from utils import save_to_stats_pkl_file, load_from_stats_pkl_file, \
    save_statistics, load_statistics, save_classification_results


class Experiment(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs=100, learning_rate=1e-03, train_data=None, val_data=None,
                 test_data=None, weight_decay_coefficient=0, use_gpu=True, continue_from_epoch=-1):
        """
        Initializes an Experiment object.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(Experiment, self).__init__()
        if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
            self.device = torch.device('cuda')  # sets device to be cuda
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
            print("use GPU")
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU

        self.experiment_name = experiment_name
        self.model = network_model
        self.model.to(self.device)  # sends the model from the cpu to the gpu
        self.model.reset_parameters()  # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, amsgrad=False,
                                    weight_decay=weight_decay_coefficient)
        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU

        if continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_acc = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch
            self.starting_epoch = continue_from_epoch
        else:
            self.starting_epoch = 0

    def run_train_iter(self, x, y):
        self.train()  
        out = self.model.forward(x)  # forward the data in the model
        loss = F.cross_entropy(input=out, target=y)  #get loss
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate 
        self.optimizer.step()  
        predicted = torch.argmax(out.data, 1)  
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  
        loss = loss.data.cpu()
        return loss, accuracy

    def run_evaluation_iter(self, x, y):
        self.eval()  # sets the system to validation mode
        out = self.model.forward(x)  # forward the data in the model
        loss = F.cross_entropy(input=out, target=y)  
        predicted = torch.argmax(out.data, 1) 
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        loss = loss.data.cpu() 
        return loss, accuracy,out.data

    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx,
                   best_validation_model_acc):
        state = dict()
        state['network'] = self.state_dict()  # save network parameter and other variables.
        state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        state['best_val_model_acc'] = best_validation_model_acc  # save current best val acc
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_acc']

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_acc": [], "train_loss": [], "val_acc": [],
                        "val_loss": []}  # initialize a dict to keep the per-epoch metrics

        if self.train_data and self.val_data:
            for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
                epoch_start_time = time.time()
                current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

                with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                    for idx, (x, y) in enumerate(self.train_data):  # get data batches
                        loss, accuracy = self.run_train_iter(x=x, y=y)  
                        current_epoch_losses["train_loss"].append(loss) 
                        current_epoch_losses["train_acc"].append(accuracy)  
                        pbar_train.update(1)
                        pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

                with tqdm.tqdm(total=len(self.val_data)) as pbar_val: 
                    for x, y in self.val_data:  
                        loss, accuracy,_ = self.run_evaluation_iter(x=x, y=y) 
                        current_epoch_losses["val_loss"].append(loss)  
                        current_epoch_losses["val_acc"].append(accuracy) 
                        pbar_val.update(1) 
                        pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
                val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])
                if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                    self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                    self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

                for key, value in current_epoch_losses.items():

                    total_losses[key].append(np.mean(value))  # get mean of all metrics of current epoch 
                save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                                stats_dict=total_losses, current_epoch=i)  # save statistics 


                out_string = "_".join(
                    ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])

                epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
                epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
                print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
                self.save_model(model_save_dir=self.experiment_saved_models,
                                # save model and best val idx and best val acc
                                model_save_name="train_model", model_idx=epoch_idx,
                                best_validation_model_idx=self.best_val_model_idx,
                                best_validation_model_acc=self.best_val_model_acc)

        if self.test_data:   
            print("Generating test set evaluation metrics")
            self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                            # load best validation model
                            model_save_name="train_model")
            current_batch_losses = {"test_acc": [], "test_loss": []}  # initialize a statistics dict
            soft_results = None
            actual_tags = None
            
            with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # init a progress bar
                for x, y in self.test_data:  # sample batch
                    loss, accuracy,results = self.run_evaluation_iter(x=x,
                                                            y=y)  # compute loss and accuracy 
                    current_batch_losses["test_loss"].append(loss)  # save test loss
                    current_batch_losses["test_acc"].append(accuracy)  # save test accuracy
                    current_batch_results = F.softmax(results,dim=1)

                    if soft_results is None:
                        soft_results = current_batch_results
                        actual_tags = y
                    else :
                        soft_results = torch.cat((soft_results, current_batch_results),0)
                        actual_tags = torch.cat((actual_tags, y),0)

                    pbar_test.update(1)  # update progress bar status
            test_losses = {key: [np.mean(value)] for key, value in
                        current_batch_losses.items()}  # save test set metrics 

            print(test_losses)
            save_classification_results(experiment_log_dir=self.experiment_logs, filename='test_results.csv',results=soft_results,
                tags=actual_tags)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                            # save test set metrics on disk in .csv format
                            stats_dict=test_losses, current_epoch=0)

            return total_losses, test_losses