
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_lcs(lcs, rows, cols):
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



