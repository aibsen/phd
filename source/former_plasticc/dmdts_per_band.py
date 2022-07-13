import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
from matplotlib.colors import LogNorm
import h5py

class DMDT_per_band:
    def __init__(self,data, xedges, yedges):
        self.ms, self.ts, self.passbands = self.splitPassBands(data)
        self.H = self.calculateDeltas(xedges,yedges)
        
    def splitPassBands(self,data):
        passbands = data['passband'].unique().tolist()
        passbands.sort()
        ms = []
        ts = []
        for passband in passbands:
            p=data[data["passband"]==passband]
            ms.append(p["flux"].values)
            ts.append(p["mjd"].values)
        return ms,ts,passbands

    def normalized_image_intensity(self,nbin,p):
        x=225*nbin/p + 0.99999
        i = int(x)
        return i

    def calculateDeltas(self,xedges,yedges):
        H = []
        for band in range(len(self.ms)):
            m_band = self.ms[band]
            t_band = self.ts[band]
            dms = [(y - x) for x, y in it.combinations(m_band, 2)]
            dts = [(y - x) for x, y in it.combinations(t_band, 2)]
            n = len(m_band)
            p = n*(n-1)/2
            H.append(self.getDMDTImg(xedges,yedges,p,dms,dts))
        return np.asarray(H)
        
    def getDMDTImg(self,xedges,yedges,p,dms,dts):
        H,xe,ye = np.histogram2d(dts, dms, bins=(xedges, yedges))
        norm_H = []
        for h in H:
            norm_h = [self.normalized_image_intensity(bin, p) for bin in h]
            norm_H.append(norm_h)
        return np.asarray(norm_H)

    def plotDeltas(self):
        fig=plt.figure()
        for col in range(0,self.H.shape[0]): 
            img = self.H[col]
            ax = fig.add_subplot(1,len(self.passbands),col+1)
            ax.set_title("band: "+str(self.passbands[col]))
            ax.imshow(img,interpolation='none', cmap=plt.cm.cool)
        plt.show()

class DMDTs_per_band:
    def __init__(self, dmdts, xedges=None, yedges=None, targets=None):
        self.dmdts = dmdts
        self.targets = targets
        self.xedges = xedges
        self.yedges = yedges
    
    def plotDeltas(self, rows, cols):
        fig=plt.figure()
        pos = 1
        ax = None
        for col in range(0,cols): 
          for row in range(0,rows):
                obj = self.dmdts[col]
                img = obj.H[row]
                if pos==1:
                    ax = fig.add_subplot(rows,cols,pos)
                else:
                    ax = fig.add_subplot(rows,cols,pos,sharex=ax, sharey=ax)
                ax.imshow(img,interpolation='none', cmap=plt.cm.plasma)
                ax.label_outer()
                pos+=1

        fig.subplots_adjust(hspace=0) 
        plt.show()

    def create_training_dataset(self, outputFile):
        print("creating")
        hf=h5py.File(outputFile,'w')
        n = self.targets.size
        X=[]
        for i,obj in enumerate(self.dmdts):
            print("getting H of obj "+str(i)+"/"+str(n))
            x = obj.H
            X.append(x)
        X=np.reshape(X,((7848,39, 40,6)))
        Y=self.targets
        X=np.asarray(X)
        print("writing X")
        hf.create_dataset('X',data=X, compression="lzf", chunks=True)
        print("writing Y")
        hf.create_dataset('Y',data=Y)
        hf.close()


 