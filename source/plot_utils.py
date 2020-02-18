
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_raw_lcs(lcs, rows, cols):
    fig, axs = plt.subplots(rows, cols,figsize=(20,10))
    #split into bands
    count = 0
    for i in np.arange(rows):
        for j in np.arange(cols):
            r=np.array(list(filter(lambda p: p["band"]=='ztfr' , lcs[count])))
            g=np.array(list(filter(lambda p: p["band"]=='ztfg' , lcs[count])))
    
            r_t = np.array(list(map(lambda p: p["time"], r)))
            r_f = np.array(list(map(lambda p: p["flux"], r)))
            r_e = np.array(list(map(lambda p: p["fluxerr"], r)))

            g_t = np.array(list(map(lambda p: p["time"], g)))
            g_f = np.array(list(map(lambda p: p["flux"], g)))
            g_e = np.array(list(map(lambda p: p["fluxerr"], g)))

            axs[i][j].errorbar(r_t, r_f, yerr= r_e,fmt='ro')
            axs[i][j].errorbar(g_t, g_f, yerr= g_e,fmt='go')
            count += 1
            
    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='flux')


def plot_raw_and_interpolated_lcs(raw_lcs, interpolated_lcs):
    rows = 4 #one for each type
    cols = 2 #one for raw data and one for interpolated data
    fig, axs = plt.subplots(rows,cols,figsize=(25,20))
    #split into bands
    for i in np.arange(rows):
        #plot raw lcs
        r=np.array(list(filter(lambda p: p["band"]=='ztfr' , raw_lcs[i])))
        g=np.array(list(filter(lambda p: p["band"]=='ztfg' , raw_lcs[i])))

        r_t = np.array(list(map(lambda p: p["time"], r)))
        r_f = np.array(list(map(lambda p: p["flux"], r)))
        r_e = np.array(list(map(lambda p: p["fluxerr"], r)))

        g_t = np.array(list(map(lambda p: p["time"], g)))
        g_f = np.array(list(map(lambda p: p["flux"], g)))
        g_e = np.array(list(map(lambda p: p["fluxerr"], g)))

        axs[i][0].errorbar(r_t, r_f, yerr= r_e,fmt='ro')
        axs[i][0].errorbar(g_t, g_f, yerr= g_e,fmt='go')
        
        #plot interpolated lcs
        axs[i][1].plot(interpolated_lcs[i][0].cpu(),'ro')
        axs[i][1].plot(interpolated_lcs[i][1].cpu(),'go')
    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='flux')


def plot_lcs_df(lc_num,data,tags):
    fig,ax=plt.subplots(lc_num, 1,figsize=(5,10))
    r=np.random.randint(0,tags.shape[0],lc_num)
    for i,random in enumerate(r):
        lc = data[data.id==tags.iloc[random].id]
        r=lc[lc.band==0]
        g=lc[lc.band==1]
        ax[i].invert_yaxis()
        ax[i].plot(r.time,r.flux,'ro-',g.time,g.flux,'go-')

def plot_train_val_acc_loss(exp_dir,n_epochs):
    results_summary = pd.read_csv(exp_dir+"summary.csv")
    fig, ax = plt.subplots(1, 2,figsize=(20,5))
    train_acc = results_summary.train_acc.values
    val_acc = results_summary.val_acc.values
    train_loss = results_summary.train_loss.values
    val_loss = results_summary.val_loss.values
    epochs = np.arange(n_epochs)
    ax[0].plot(epochs,train_acc)
    ax[0].plot(epochs,val_acc)
    ax[1].plot(epochs,train_loss)
    ax[1].plot(epochs,val_loss)
    ax[0].set(xlabel='epochs', ylabel='accuracy')
    ax[1].set(xlabel='epochs', ylabel='loss')