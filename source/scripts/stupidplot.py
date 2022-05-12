
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
import os, sys
import scipy.stats as stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_cm(ax,true_targets, predictions, normalized=True,colormap=None):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    cm=confusion_matrix(true_targets,predictions)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if colormap is not None:
        im = ax.imshow(cm, interpolation= 'nearest', cmap=colormap)
    else : 
        im = ax.imshow(cm, interpolation= 'nearest', cmap=plt.cm.PuBu)
    # names = ["snIa"," ","snIb/c"," ","snIIn"," ","snIIP"]
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
    # ax.set_xticklabels([''] + names)
    # ax.set_yticklabels([''] + names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")



results_dir = "../../results/"
exp_name0 = "plasticc_vanilla_f-1"
exp_name1 = "plasticc_vanilla_sk"
exp_name2 = "plasticc_vanilla"

#stupid plot 1
# file_name = "/result_outputs/test_results.csv"

# fig, ax = plt.subplots(1,3)


# for i, exp_name in enumerate([exp_name0,exp_name1,exp_name2]):
#     df = pd.read_csv(results_dir+exp_name+file_name)
#     targets = list(df.targets)
#     predictions = list(df.predictions)
#     plot_cm(ax[i], targets, predictions)


# plt.show()

#stupid plot 2
# fig, ax = plt.subplots(1,3)
ax0 = plt.subplot(131)
ax1 = plt.subplot(132,sharey=ax0)
ax2 = plt.subplot(133,sharey=ax1)
ax = [ax0,ax1,ax2]

def plot_epochs(ax,train_loss,train_f1,epoch_train, val_loss,val_f1,epoch_val):
    ax.plot(epoch_train,train_loss)
    ax.plot(epoch_train,train_f1)
    ax.plot(epoch_val,val_loss)
    ax.plot(epoch_val,val_f1)
    # ax.label_outer()



file_name_train = "/result_outputs/training_summary.csv"
file_name_val = "/result_outputs/validation_summary.csv"

for i, exp_name in enumerate([exp_name0,exp_name1,exp_name2]):

    df_train = pd.read_csv(results_dir+exp_name+file_name_train)
    df_val = pd.read_csv(results_dir+exp_name+file_name_val)
    train_loss = list(df_train.train_loss)
    train_f1 = list(df_train.train_f1)
    val_loss = list(df_val.val_loss)
    val_f1 = list(df_val.val_f1)
    epoch_train = list(df_train.epoch)
    epoch_val = list(df_val.epoch)
    
    plot_epochs(ax[i], train_loss,train_f1,epoch_train, val_loss, val_f1,epoch_val)
    # plt.ylim(0,5)
    # for ax in fig.get_axes():
    #     ax.label_outer()
# fig.align

plt.show()

