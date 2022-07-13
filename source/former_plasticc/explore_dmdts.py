import pandas as pd
import numpy as np
import time
from dmdts_per_band import DMDT_per_band, DMDTs_per_band
from utils import load_data, random_objs_per_class, get_classes


def plot_dmdts_per_band(data, ids, dtrandes,dmranges):
    D = []
    for id in ids:
        lc = data[data["obj_id"]==id[0]]
        lc = lc.sort_values(by=['mjd'])
        dmdt_per_band = DMDT_per_band(lc[["obj_id",'flux','passband',"mjd"]],dtranges,dmranges)
        # dmdt_per_band.plotDeltas()
        # break
        D.append(dmdt_per_band)
    
    dmdts_per_band=DMDTs_per_band(D,dtranges,dmranges)
    dmdts_per_band.plotDeltas(6,len(ids))


def create_trainning_dataset(data,metadata,ids,dtranges,dmranges):
    D = []
    targets=[]
    n=len(ids)
    t0 = time.time()
    for i,id in enumerate(ids):
        print("getting dmdt for obj "+str(i)+"/"+str(n))
        lc = data[data["obj_id"]==id]
        # lc = lc.sort_values(by=['mjd'])
        target = metadata[metadata["object_id"]==id]["target"].values[0]
        dmdt_per_band = DMDT_per_band(lc[["obj_id",'flux','passband',"mjd"]],dtranges,dmranges)
        D.append(dmdt_per_band)
        targets.append(target)
        
    dmdts_per_band=DMDTs_per_band(np.asarray(D),dtranges,dmranges,np.asarray(targets))
    # dmdts_per_band.to_pickle("dmdts.pkl",compression="gzip")
    print("saving to file...")
    dmdts_per_band.create_training_dataset("dmdts_per_band.h5")
    t1 = time.time()
    total_n = t1-t0
    print("total execution time: "+str(total_n))

if __name__ == "__main__":
    [data, metadata, ids] = load_data("training_set.csv", "retagged_training_set_metadata.csv")
    classes = get_classes(metadata)
    #get just one random obj for now
    rand_id = random_objs_per_class(1,classes,metadata)
    max_flux_delta = data["flux"].max()-data["flux"].min()
    max_mjd_delta = data["mjd"].max()-data["mjd"].min()

    dmranges = [-max_flux_delta,
   -10000,-5000,-3000,-1000,-800,-600,
   -500,-300,-200,-100,-80,-60,-50,
    -40,-30,-20,-10,-5,-2,0,2,5,10,20,30,40,
    50,60,80,100,200,300,500,
    600,800,1000,3000,5000,10000,
     max_flux_delta]
    dtranges =  np.arange(0, max_mjd_delta,max_mjd_delta/40)

    # plot_dmdts_per_band(data,rand_id,dtranges,dmranges)
    create_trainning_dataset(data, metadata, ids, dtranges,dmranges)

