import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
from matplotlib.colors import LogNorm

class DMDT:
    def __init__(self, ts, ms, xedges=None, yedges=None, name=""):
        self.name = name
        self.ms = ms
        self.ts = ts
        self.H, self.dms, self.dts, self.p = self.calculateDeltas(xedges,yedges)
        

    def normalized_image_intensity(self,nbin,p):
        x=225*nbin/p + 0.99999
        i = int(x)
        return i

    def calculateDeltas(self,xedges,yedges):
        dms = [(y - x) for x, y in it.combinations(self.ms, 2)]
        dts = [(y - x) for x, y in it.combinations(self.ts, 2)]
        n = len(self.ms)
        p = n*(n-1)/2
        if (xedges != None and yedges != None):
            H = self.getDMDTImg(xedges,yedges,p)
            return H,dms,dts,p
        else:
            return None, dms, dts,p
        
    def getDMDTImg(self,xedges,yedges,p):

        H,xe,ye = np.histogram2d(self.dts, self.dms, bins=(xedges, yedges))
        norm_H = []
        for h in H:
            norm_h = [self.normalized_image_intensity(bin, p) for bin in h]
            norm_H.append(norm_h)
        return np.asarray(norm_H)

    def plotDeltas(self):
        plt.imshow(self.H, interpolation='none', cmap=plt.cm.cool)
        plt.colorbar()
        plt.show()

class DMDTs:
    def __init__(self, dmdts, xedges=None, yedges=None):
        self.dmdts = dmdts
        self.xedges = xedges
        self.yedges = yedges
    
    def plotDeltas(self, rows, columns):
        fig=plt.figure()
        dmdts = self.dmdts
        c = 0
        for col in range(0,columns): 
            pos = col+1
            for row in range(0,rows):
                img = dmdts[c].H
                if img == None:
                    # print(dmdts[c].p)
                    # print("p")
                    img = dmdts[c].getDMDTImg(self.xedges, self.yedges,dmdts[c].p) 
                ax = fig.add_subplot(rows,columns,pos)
                ax.set_title(dmdts[c].name)
                ax.imshow(img,interpolation='none', cmap=plt.cm.cool)
                pos = pos + columns
                c=c+1
        plt.show()
