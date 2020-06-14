import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import confusion_matrix
# from plot_utils import *


def plot_train_val_acc_loss(exp_dir,n_epochs):
    results_summary = pd.read_csv(exp_dir+"/result_outputs/summary.csv")
    fig, ax = plt.subplots(1, 2,figsize=(20,5))
    train_acc = results_summary.train_f1.values
    val_acc = results_summary.val_f1.values
    train_loss = results_summary.train_loss.values
    val_loss = results_summary.val_loss.values
    epochs = np.arange(n_epochs)
    ax[0].plot(epochs,train_acc)
    ax[0].plot(epochs,val_acc)
    ax[1].plot(epochs,train_loss)
    ax[1].plot(epochs,val_loss)
    ax[0].set(xlabel='epochs', ylabel='accuracy')
    ax[1].set(xlabel='epochs', ylabel='loss')
    plt.show()

def plot_cms(files, rows, cols):
    fig, axs = plt.subplots(rows, cols)
    tags_predictions = []
    
    for f in files:
        test_results = pd.read_csv(f)
        tt = test_results.true_tags.values
        pt = test_results.predicted_tags.values
        tags_predictions.append([tt,pt])

    for i,ax in enumerate(axs):
        tt = tags_predictions[i][0]
        pt = tags_predictions[i][1]
        plot_cm(ax,tt,pt)

    titles = ["FCN", "ResNet", "RNN", "RNN with self-attention"]
    for i,ax in enumerate(fig.get_axes()):
        ax.label_outer()
        ax.title.set_text(titles[i])

    plt.show()

def plot_cm(ax,true_targets, predictions, normalized=True):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    cm=confusion_matrix(true_targets,predictions)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm, interpolation= 'nearest', cmap=plt.cm.Blues)
    names = ["snIa"," ","snIb/c"," ","snIIn"," ","snIIP"]
    # namesy = ["snIa"," ","snIb/c"," ","snIIn"," ","snIIP"]
    # namesx = ["snIa","snIb/c","snIIn","snIIP"]
    fmt ='.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalized:
                ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            else :
                ax.text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xticklabels([''] + names)
    ax.set_yticklabels([''] + names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")

def plot_train_val_f1s(filenames,n_epochs,rows,cols):
    fig, ax = plt.subplots(rows, cols)
    titles = ["FCN", "ResNet", "RNN", "RNN with self-attention"]
    for i,f in enumerate(filenames):
        results_summary = pd.read_csv(f)
        train_f1 = results_summary.train_f1.values
        val_f1 = results_summary.val_f1.values
        epochs = np.arange(n_epochs)
        ax[i].plot(epochs,train_f1,color='#636389')
        ax[i].plot(epochs,val_f1,color='#eb8c00')
        ax[i].set(xlabel='epochs')
        ax[i].label_outer()
        ax[i].title.set_text(titles[i])
        best_epoch = val_f1.argmax()
        best_f1_score = '%.3f' % val_f1.max()
        best_epoch_label="best F1-score:{} at epoch {}".format(best_f1_score,best_epoch)
        ax[i].axvline(x=best_epoch,ls='--', label = best_epoch_label, color= "#e96342")
        ax[i].legend()
    # custom_xlim = (0, 100)
    fig.text(0.06, 0.5, 'F1-Score', ha='center', va='center', rotation='vertical')
    custom_ylim = (0, 1)
    # plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
    plt.setp(ax, ylim=custom_ylim)
    plt.show()

results_dir = "../../results/"
exp=2
part =1
exp_names = list(map(lambda x: x.format(exp,part),["exp{}_p{}_fcn", "exp{}_p{}_resnet", "exp{}_p{}_gru", "exp{}_p{}_grusa"]))
filenames = list(map(lambda x: results_dir+x+"/result_outputs/test_results.csv",exp_names))
print(filenames) 
plot_cms(filenames,1,len(filenames))
# filenames2 = list(map(lambda x: results_dir+x+"/result_outputs/summary.csv",exp_names))
# plot_train_val_f1s(filenames2,100,4,1)
