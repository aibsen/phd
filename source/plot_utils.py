
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

def plot_cms(files, rows, cols,color=None):
    fig, axs = plt.subplots(rows, cols)
    tags_predictions = []
    
    for f in files:
        test_results = pd.read_csv(f)
        print(f)
        print(test_results.shape)
        tt = test_results.true_tags.values
        pt = test_results.predicted_tags.values
        tags_predictions.append([tt,pt])


    for i,ax in enumerate(axs):
        tt = tags_predictions[i][0]
        pt = tags_predictions[i][1]
        if color:
            colormap = make_colormap_from_color(color)
            plot_cm(ax,tt,pt,colormap=colormap)
        else:
            plot_cm(ax,tt,pt)

    titles = ["FCN", "ResNet", "RNN", "RNN with self-attention"]
    for i,ax in enumerate(fig.get_axes()):
        ax.label_outer()
        ax.title.set_text(titles[i])

    plt.show()

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

def plot_best_val_cm(target, prediction, normalized=True, colormap=None, names=plasticc_names,verbose=False, save=False, output_file=None):
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
    # fig.figsize(100,100)

    if verbose:
        plt.show()
    
    if save and output_file:
        plt.savefig(output_file)
    elif save:
        print("Need output_file argument")

def plot_best_cms(results_dir = "../../results/",exp=2,part=1,count=3):
    
    exp_names = list(map(lambda x: x.format(exp,part),["exp{}_p{}_fcn", "exp{}_p{}_resnet", "exp{}_p{}_gru", "exp{}_p{}_grusa"]))
    count = 20
    test_results="test_results_new_count{}.csv".format(count)
    test_summary="test_results_new_summary{}.csv".format(count)
    filenames = list(map(lambda x: SeededExperiment(results_dir+x).get_best_results((test_results, test_summary)[0])[2],exp_names))
    plot_cms(filenames,1,len(filenames))

def plot_histograms(metric="f1",results_dir = "../../results/",exp=2,part=1,count=3):
    
    exp_names = list(map(lambda x: x.format(exp,part),["exp{}_p{}_fcn", "exp{}_p{}_resnet", "exp{}_p{}_gru", "exp{}_p{}_grusa"]))
    summary_filename = "test_results_new_summary{}.csv".format(count)
    # summary_filename="validation_summary.csv"
    metrics = np.asarray(list(map(lambda x: SeededExperiment(results_dir+x).get_all_metrics(metric=metric,summary_filename=summary_filename), exp_names)))
    metrics = np.swapaxes(metrics, 0,1)
    # colors =["#877cb2","#e37b5f","#aa4773","#47a5aa"]
    colors= ["#c02878","#20c8b8","#e8a000","#104890"]
    names = ["FCN", "ResNet", "RNN", "RNN-SA"]
    n, bins, patches = plt.hist(metrics, 10, density=False,color=colors,alpha=0.7)
    for i in np.arange(len(exp_names)):
        mu = np.mean(metrics[:,i])
        sigma = np.std(metrics[:,i])
        x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
        dx = bins[1] - bins[0]
        scale = metrics.shape[0]*dx
        mustr = "%.3f" % mu
        sigmastr = "%.3f" % sigma
        label = ', $\mu={},\ \sigma={}$'.format(mustr,sigmastr)
        plt.plot(x, stats.norm.pdf(x, mu, sigma)*scale, color=colors[i],label=names[i]+label, alpha=0.8)
    # plt.xlim(right=1)
    # plt.xlim(left=0)
    # plt.xticks([0.1,0.2,0.3,0.40,0.5,0.6,0.7,0.8,0.9,1])
    plt.yticks(np.arange(0,30,5))
    plt.ylim(top=26)
    plt.legend()
    plt.show()

def plot_cumulative(metric="f1",results_dir = "../../results/",exp=2,part=1,count=3):

    exp_names = list(map(lambda x: x.format(exp,part),["exp{}_p{}_fcn", "exp{}_p{}_resnet", "exp{}_p{}_gru", "exp{}_p{}_grusa"]))
    summary_filename = "test_results_new_summary{}.csv".format(count)
    # summary_filename="validation_summary.csv"
    metrics = np.asarray(list(map(lambda x: SeededExperiment(results_dir+x).get_all_metrics(summary_filename=summary_filename), exp_names)))
    metrics = np.swapaxes(metrics, 0,1)
    # colors =["#877cb2","#e37b5f","#aa4773","#47a5aa"]
    colors= ["#c02878","#20c8b8","#e8a000","#104890"]
    names = ["FCN", "ResNet", "RNN", "RNN-SA"]
    n, bins, patches = plt.hist(metrics, 10,color=colors,alpha=0.8,cumulative=1,label=names, histtype='step')
    
    # plt.xlim(right=1)
    # plt.xlim(left=0)
    # plt.xticks([0.1,0.2,0.3,0.40,0.5,0.6,0.7,0.8,0.9,1])
    plt.yticks(np.arange(0,30,5))
    plt.ylim(top=26)
    plt.legend(loc=2)
    plt.show()

def plot_best_cms_rapid():
    test_summary="validation_summary.csv"
    test_results="validation_results.csv"
    filenames = list(map(lambda x: SeededExperiment(exp_dir+x).get_best_results((test_results, test_summary)[0])[2],models))
    print(filenames)
    plot_cms(filenames,1,len(filenames))
    plt.show()

# plot_best_cms_rapid()

def plot_histograms_rapid(metric="f1",results_dir = "../../results/exp2_rapid_p1_{}"):
    print(models)
    exp_names = list(map(lambda x: results_dir.format(x),models))
    print(exp_names)
    summary_filename = "test_summary.csv"
    # summary_filename="validation_summary.csv"
    metrics = np.asarray(list(map(lambda x: SeededExperiment(x).get_all_metrics(metric=metric,summary_filename=summary_filename), exp_names)))
    metrics = np.swapaxes(metrics, 0,1)
    # # colors =["#877cb2","#e37b5f","#aa4773","#47a5aa"]
    colors= ["#c02878","#20c8b8","#e8a000","#104890"]
    names = ["FCN", "ResNet", "RNN", "RNN-SA"]
    n, bins, patches = plt.hist(metrics, 10, density=False,color=colors,alpha=0.7)
    for i in np.arange(len(exp_names)):
        mu = np.mean(metrics[:,i])
        sigma = np.std(metrics[:,i])
        x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
        dx = bins[1] - bins[0]
        scale = metrics.shape[0]*dx
        mustr = "%.3f" % mu
        sigmastr = "%.3f" % sigma
        label = ', $\mu={},\ \sigma={}$'.format(mustr,sigmastr)
        plt.plot(x, stats.norm.pdf(x, mu, sigma)*scale, color=colors[i],label=names[i]+label, alpha=0.8)
    # plt.xlim(right=1)
    # plt.xlim(left=0)
    # plt.xticks([0.1,0.2,0.3,0.40,0.5,0.6,0.7,0.8,0.9,1])
    plt.yticks(np.arange(0,30,5))
    plt.ylim(top=26)
    plt.legend()
    plt.show()