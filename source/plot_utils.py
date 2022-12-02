
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
import os, sys
import scipy.stats as stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from seeded_experiment import SeededExperiment


np.set_printoptions(threshold=np.inf)
plasticc_type_dict = {
    '90':'SN-Ia',
    '67':'SN-Ia-91BG', 
    '52':'SN-Iax',  
    '42':'SN-II', 
    '62':'SN-Ib/c',
    '95':'SLSN',
    '15':'TDE',
    '64':'KN',
    '88':'AGN',
    '92':'RRL',
    '65':'M-dwarf',
    '16':'EB',
    '53':'Mira',
    '6':'uLens-Single',
}
plasticc_names = [plasticc_type_dict[k] for k in plasticc_type_dict]

def make_colormap_from_color(color):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """

    c = mcolors.ColorConverter().to_rgb
    seq=[c('white'),c(color)]
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

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


def plot_raw_and_interpolated_lcs(raw_lcs, interpolated_lcs, units='flux'):
    rows = 1 
    cols = 2 #one for raw data and one for interpolated data
    fig, axs = plt.subplots(rows,cols,figsize=(25,20))
    #split into bands
    raw_r = raw_lcs[raw_lcs.passband==0]
    raw_g = raw_lcs[raw_lcs.passband==1]


    interp_r = interpolated_lcs[interpolated_lcs.passband==0]
    interp_g = interpolated_lcs[interpolated_lcs.passband==1]

    if units == 'flux':
        axs[0].errorbar(raw_r.mjd, raw_r.flux, yerr=raw_r.flux_err)
        axs[0].errorbar(raw_g.mjd, raw_g.flux, yerr=raw_g.flux_err)

        axs[1].errorbar(interp_r.mjd, interp_r.flux, yerr=interp_r.flux_err)
        axs[1].errorbar(interp_g.mjd, interp_g.flux, yerr=interp_g.flux_err)
    elif units == 'mag':
        axs[0].errorbar(raw_r.mjd, raw_r.magpsf, yerr=raw_r.sigmagpsf)
        axs[0].errorbar(raw_g.mjd, raw_g.magpsf, yerr=raw_g.sigmagpsf)
        axs[0].invert_yaxis()

        axs[1].errorbar(interp_r.mjd, interp_r.magpsf, yerr=interp_r.sigmagpsf)
        axs[1].errorbar(interp_g.mjd, interp_g.magpsf, yerr=interp_g.sigmagpsf)
        axs[1].invert_yaxis()
    else:
        print("Units needs to be either flux or mag")
    plt.show()
    # for i in np.arange(rows):
    #     #plot raw lcs
    #     r=np.array(list(filter(lambda p: p["passband"]=='0' , raw_lcs[i])))
    #     g=np.array(list(filter(lambda p: p["passband"]=='1' , raw_lcs[i])))
    #     print(r)

    #     r_t = np.array(list(map(lambda p: p["time"], r)))
    #     r_f = np.array(list(map(lambda p: p["flux"], r)))
    #     r_e = np.array(list(map(lambda p: p["fluxerr"], r)))

    #     g_t = np.array(list(map(lambda p: p["time"], g)))
    #     g_f = np.array(list(map(lambda p: p["flux"], g)))
    #     g_e = np.array(list(map(lambda p: p["fluxerr"], g)))

    #     axs[i][0].errorbar(r_t, r_f, yerr= r_e,fmt='ro')
    #     axs[i][0].errorbar(g_t, g_f, yerr= g_e,fmt='go')

    #     #plot interpolated lcs
    #     axs[i][1].plot(interpolated_lcs[i][0].cpu(),'ro')
    #     axs[i][1].plot(interpolated_lcs[i][1].cpu(),'go')
    # for ax in axs.flat:
    #     ax.set(xlabel='time', ylabel='flux')


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
    plt.show()

def plot_train_val_f1s(filenames,n_epochs,rows,cols):
    fig, ax = plt.subplots(rows, cols)
    # color = ["#8bcc78", "#29aae1"
    # color = ["#8bcc78", "#29aae1","#ff7f50","#7030a0"]
    # color2 = ["#1d6309", "#0d4b92", "#a92e01","#301545"]
    green="#8bcc78"
    blue="#29aae1"
    pink = "#e05a94"
    titles = ["FCN", "ResNet", "RNN", "RNN with self-attention"]
    for i,f in enumerate(filenames):
        results_summary = pd.read_csv(f)
        train_f1 = results_summary.train_f1.values
        val_f1 = results_summary.val_f1.values
        epochs = np.arange(n_epochs)
        ax[i].plot(epochs,train_f1,color=green, label="F1-score for the training set")
        ax[i].plot(epochs,val_f1,color=blue, label="F1-score for the validation set")
        ax[i].set(xlabel='epochs')
        ax[i].label_outer()
        ax[i].title.set_text(titles[i])
        best_epoch = val_f1.argmax()
        best_f1_score = '%.3f' % val_f1.max()
        best_epoch_label="best F1-score:{} at epoch {}".format(best_f1_score,best_epoch)
        ax[i].axvline(x=best_epoch,ls="--", label = best_epoch_label, color=pink)
        ax[i].legend()
    # custom_xlim = (0, 100)
    fig.text(0.06, 0.5, 'F1-Score', ha='center', va='center', rotation='vertical')
    custom_ylim = (0, 1)
    # plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
    plt.setp(ax, ylim=custom_ylim)
    plt.show()

def autolabel(ax,rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_sns_by_type(type_dict,metadata_file,colors=None):
    """This function recieves a metadata file as provided by tns and a dictionary of types
    and subtypes and plots objects per subtype"""
    real_sns = pd.read_csv(metadata_file)

    by_type = real_sns.groupby(["Obj. Type"]).count()
    by_type["Type"] = by_type.index.values
    for k,v in sn_dict.items():
        by_type.loc[by_type["Type"].str.contains(k),"Type"] = v

    by_parent_type = by_type.groupby(["Type"]).sum()
    print(by_parent_type)

    fig, ax = plt.subplots()
    rects = ax.bar(by_parent_type.index.values,by_parent_type.ID.values)
    print(rects)
    if colors:
        assert len(colors) == len(rects)
        for rect,color in zip(rects,colors):
            rect.set_color(color)

    plt.xticks(rotation=45)
    autolabel(ax,rects)
    plt.show()

def plot_sns_by_date(metadata_file,color=None):

    """This function recieves a metadata file as provided by tns and a dictionary of types
    and subtypes and plots objects per subtype"""
    real_sns = pd.read_csv(metadata_file)
    real_sns["by_month"]=real_sns["Discovery Date (UT)"]
    real_sns["by_month"] = real_sns["by_month"].str.split(" ").apply(lambda x : x[0])
    real_sns["by_month"] = real_sns["by_month"].str.split("-").apply(lambda x : x[0]+"-"+x[1])
    by_month = real_sns.groupby(["by_month"]).count()

    fig, ax = plt.subplots()
    if color:
        rects = ax.bar(by_month.index.values,by_month.ID.values,color=color)
    else:
        rects = ax.bar(by_month.index.values,by_month.ID.values)

    plt.xticks(rotation=45)
    autolabel(ax,rects)
    plt.show()

def plot_cms(files, rows, cols,color=None, 
    subtitles = ["FCN", "ResNet", "RNN", "RNN with self-attention"],
    classes = ["snIa","snIb/c","snIIn","snIIP"],
    title = '',
    verbose = True,
    save = False,
    output_file = None):
    
    fig, axs = plt.subplots(rows, cols)
    # ,sharex=True,sharey=True)
    tags_predictions = []
    
    for f in files:
        test_results = pd.read_csv(f)
        print(f)
        print(test_results.shape)
        tt = test_results.target.values
        pt = test_results.prediction.values
        tags_predictions.append([tt,pt])

    axs=axs.flatten()
    print(axs)
    for i,ax in enumerate(axs):
        print(ax)
        tt = tags_predictions[i][0]
        pt = tags_predictions[i][1]
        if color:
            colormap = make_colormap_from_color(color)
            draw_cm(ax,tt,pt,colormap=colormap,names=classes)
        else:
            draw_cm(ax,tt,pt,names=classes)
    
    for i,ax in enumerate(fig.get_axes()):
        # ax.label_outer()
        ax.title.set_text(subtitles[i])
    
    # for i,ax in enumerate(fig.get_axes()[0:3]):
    #     # ax.label_outer()
    #     ax.set_title(subtitles[i],size=20)

    # nns =["Self-Attention", "GRU","FCN", "ResNet"]
    # for i,ax in enumerate(fig.get_axes()):
    #     if i%3==0:
    #         ax.set_ylabel(nns[int(i/3)],size=20)


    fig.supxlabel('Predicted class',size=14)
    fig.supylabel('True class',size=14)
    # fig.suptitle(title)

    if save and output_file:
        # fig.set_figheight(9*rows)
        fig.set_figheight(7*rows)
        # fig.set_figwidth(9*cols)
        fig.set_figwidth(7*cols)
        # fig.set_dpi(100)
        plt.tight_layout()
        plt.savefig(output_file)
    elif save:
        print("Need output_file argument")
    if verbose:
        plt.show()


def draw_cm(ax,target, prediction, normalized=True, colormap=None, names=plasticc_names):
    # print(names)
    # fig,ax = plt.subplots(1,1)
    cm=confusion_matrix(target,prediction)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if colormap is not None:
        im = ax.imshow(cm, interpolation= 'nearest', cmap=colormap)
    else : 
        im = ax.imshow(cm, interpolation= 'nearest', cmap=plt.cm.Purples)
    fmt ='.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalized:
                ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",size=14)
            else :
                ax.text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xticks(range(0,len(names)))
    ax.set_yticks(range(0,len(names)))
    ax.set_xticklabels(names,size=14)
    ax.set_yticklabels(names,size=14)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

def plot_cm(target, prediction, normalized=True, colormap=None, names=plasticc_names,
    verbose=False, save=False, output_file=None):

    fig,ax = plt.subplots(1,1)
    draw_cm(ax,target,prediction,normalized=normalized,names=names,colormap=colormap)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    # print(names)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.set_dpi(100)
    # fig.figsize(100,100)
    # size = fig.get_size_inches()
    # print(size)
    if verbose:
        plt.show()
    
    if save and output_file:
        # print(size)
        # print(fig.dpi)
        plt.savefig(output_file)
    elif save:
        print("Need output_file argument") 

def plot_best_val_cm(target, prediction, normalized=True, colormap=None, names=plasticc_names,verbose=False, save=False, output_file=None):
    # print(names)
    fig,ax = plt.subplots(1,1)
    cm=confusion_matrix(target,prediction)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if colormap is not None:
        im = ax.imshow(cm, interpolation= 'nearest', cmap=colormap)
    else : 
        im = ax.imshow(cm, interpolation= 'nearest', cmap=plt.cm.BuPu)
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
    ax.set_xticks(range(0,len(names)))
    ax.set_yticks(range(0,len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    # print(names)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.set_dpi(100)
    # fig.figsize(100,100)
    # size = fig.get_size_inches()
    # print(size)
    if verbose:
        plt.show()
    
    if save and output_file:
        # print(size)
        # print(fig.dpi)
        plt.savefig(output_file)
    elif save:
        print("Need output_file argument")
    return cm


def plot_reconstruction(original,reconstructed,l):
    fig,ax = plt.subplots(1,1)
    ax.scatter(np.arange(l),original[0,-l:],label='original',color='r',marker='+')
    ax.scatter(np.arange(l),original[1,-l:],label='original',color='g', marker='+')
    ax.scatter(np.arange(l),reconstructed[0,-l:],label='reconstructed',color='r')
    ax.scatter(np.arange(l),reconstructed[1,-l:],label='reconstructed',color='g')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

def plot_reconstructions_per_epoch(random_samples, step,best_epoch,output_fn):
    random_samples = [r for r in random_samples if r[5]<= best_epoch]
    length = len(random_samples)
    cols = int(step)
    rows = int(np.ceil(length/cols))
    i = 0
    # print(len(random_samples))
    if length ==1:
        fig, ax = plt.subplots(1, 1)
        #original, reconstructed, length, id, class, epoch 
        original = random_samples[0][0]
        reconstructed = random_samples[0][1]
        l = random_samples[0][2] -1
        id = random_samples[0][3]
        c = random_samples[0][4]
        epoch = random_samples[0][5]
        ax.scatter(np.arange(l),original[0,-l:],linewidths=1.0, label='original r band', color='r', marker='+')
        ax.scatter(np.arange(l),original[1,-l:],linewidths=1.0, label='original g band', color='g', marker='+')
        ax.scatter(np.arange(l),reconstructed[0,-l:],linewidths=1.0, label='reconstructed r band', color='r', marker='.')
        ax.scatter(np.arange(l),reconstructed[1,-l:],linewidths=1.0,  label='reconstructed g band', color='g', marker='.')
        ax.set_title("epoch {}".format(epoch+1),fontsize='medium')
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

        ax.invert_yaxis()
        handles, labels = ax.get_legend_handles_labels()

    elif rows==1:
        fig, ax = plt.subplots(1, length)
        for col in np.arange(length):
            #original, reconstructed, length, id, class, epoch 
            original = random_samples[col][0]
            reconstructed = random_samples[col][1]
            l = random_samples[col][2] -1
            id = random_samples[col][3]
            c = random_samples[col][4]
            epoch = random_samples[col][5]
            ax[col].scatter(np.arange(l),original[0,-l:],linewidths=1.0, label='original r band', color='r', marker='+')
            ax[col].scatter(np.arange(l),original[1,-l:],linewidths=1.0, label='original g band', color='g', marker='+')
            ax[col].scatter(np.arange(l),reconstructed[0,-l:],linewidths=1.0, label='reconstructed r band', color='r', marker='.')
            ax[col].scatter(np.arange(l),reconstructed[1,-l:],linewidths=1.0,  label='reconstructed g band', color='g', marker='.')
            ax[col].set_title("epoch {}".format(epoch+1),fontsize='medium')

            ax[col].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

            ax[col].invert_yaxis()
        handles, labels = ax[0].get_legend_handles_labels()

    else:    
        fig, ax = plt.subplots(rows, cols)
        for row in np.arange(rows):
            for col in np.arange(cols):
                if i < len(random_samples):
                    #original, reconstructed, length, id, class, epoch 
                    original = random_samples[i][0]
                    reconstructed = random_samples[i][1]
                    l = random_samples[i][2] -1
                    id = random_samples[i][3]
                    c = random_samples[i][4]
                    epoch = random_samples[i][5]
                    ax[row][col].scatter(np.arange(l),original[0,-l:],linewidths=1.0, label='original r band', color='r', marker='+')
                    ax[row][col].scatter(np.arange(l),original[1,-l:],linewidths=1.0, label='original g band', color='g', marker='+')
                    ax[row][col].scatter(np.arange(l),reconstructed[0,-l:],linewidths=1.0,label='reconstructed r band', color='r', marker='.')
                    ax[row][col].scatter(np.arange(l),reconstructed[1,-l:],linewidths=1.0, label='reconstructed g band', color='g', marker='.')
                    ax[row][col].set_title("epoch {}".format(epoch+1),fontsize='medium')

                    ax[row][col].tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off

                    ax[row][col].invert_yaxis()
                else:
                    fig.delaxes(ax[row][col])

                i+=1
    
        handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    fig.set_figheight(2*rows)
        # fig.set_figwidth(9*cols)
    fig.set_figwidth(3.5*cols)
        # fig.set_dpi(100)
    # fig.set_figheight(12)
    # fig.set_figwidth(12)
    fig.set_dpi(100)
    # fig.suptitle("Light curve reconstruction per epoch")
    plt.tight_layout()

    plt.savefig(output_fn)
    # plt.show()
            