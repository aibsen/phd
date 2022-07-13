import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
from matplotlib.colors import LogNorm

class HeatMap:
    def __init__(self, ts, ms, xedges=None, yedges=None, name=""):
        self.name = name
        self.ms = ms
        self.ts = ts
        
    def plotHM(self):
        # plt.plot(self.ts,self.ms)
        plt.imshow(self.ts[np.newaxis,:], cmap="plasma", aspect="auto")
        plt.show()

    def plotSeparateHM(self):
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, 7)]
        fig=plt.figure(figsize=[2.5,3.5],frameon=False)
        l = len(self.ms)
        ax = fig.add_subplot(l,1,1)
        fig.subplots_adjust(hspace=0)

        ax.plot(self.ts[0],self.ms[0],color =colors[0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for i in range(1,len(self.ms)):
            ax = fig.add_subplot(len(self.ms),1,i+1, sharey=ax, sharex = ax)
            ax.plot(self.ts[i],self.ms[i],color =colors[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig("features1/"+self.name+".png", bbox_inches='tight')
        plt.close("all")
        # fig.canvas.draw()
        # ncols, nrows = fig.canvas.get_width_height()
        # ble = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
        # print(ble)
        # print(ble.shape)


class HeatMaps:
    def __init__(self, lines):
        self.lines = lines
    
    def plotHM(self, rows, columns):
        fig=plt.figure()
        lines = self.lines
        c = 0
        aux_ax = None
        for col in range(0,columns): 
            pos = col+1
            for row in range(0,rows):
                if(col == 0 and row == 0):
                    aux_ax = fig.add_subplot(rows,columns,pos)
                ax = fig.add_subplot(rows,columns,pos,sharey=aux_ax)   
                # ax = fig.add_subplot(rows,columns,pos)   
                ax.set_title(lines[c].name)
                ax.imshow(lines[c].ms[np.newaxis,:], aspect="auto", cmap=plt.get_cmap("inferno"))
                pos = pos + columns
                c=c+1
        plt.show()
