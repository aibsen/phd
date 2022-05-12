from astropy.io import fits
from astropy.table import Table
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess_data_utils import *

data_dir = "../../data/rapid_data/ZTF_20190512/"
models = ["01","02","03","12","13","14","41","43","51","60","61","62","63","64"]
tags = {"01":0,"02":2,"03":1,"12":2,"13":1,"14":2,"41":3,"43":4,"51":5,"60":6,"61":7,"62":8,"63":9,"64":10}

for i,model in enumerate(models):
    print("model "+model)
    c=0
    data = []
    labels = []
    ids = []

    for f in np.arange(1,41):
        print("file "+str(f)+"/40")
        f_str = str(f) if f >=10 else "0"+str(f)
        filename = data_dir+"ZTF_MSIP_MODEL{}/ZTF_MSIP_NONIaMODEL0-00{}_PHOT.FITS".format(model,f_str)

        dat = Table.read(filename, format='fits')
        df = dat.to_pandas()
        # print(df.groupby(["FIELD"]).count())
        df["FIELD"]=df["FIELD"].apply(lambda x: str(x).strip()[2:-1].strip())#getting rid of formatting error
        df["FLT"]=df["FLT"].apply(lambda x: str(x).strip()[2:-1].strip())#getting rid of formatting error

        # transform fluxes to magnitudes
        df = df[df['FLUXCAL']>=0]
        df['flux'] = flux_to_abmag(df["FLUXCAL"],zp=df["ZEROPT"])
        # make sure there are points in both bands
        group_by_id_band = df.groupby(['FIELD','FLT'])['MJD'].agg(['count']).rename(columns = lambda x : 'time_' + x).reset_index()
        # print(group_by_id_band.head())
        group_by_id_band = group_by_id_band.groupby(['FIELD']).count()
        # print(group_by_id_band.head())
        ids_enough_point_count = group_by_id_band[group_by_id_band.time_count==2]
        # print(ids_enough_point_count.head())
        usable_ids = list(set(ids_enough_point_count.index.values))
        df = df[df.FIELD.isin(usable_ids)]
        # print(df.groupby(["FIELD"]).count())
        # convert bands to the code's sim uses (r=0, g=1)
        df.loc[df.FLT=="r",'FLT'] = 0
        df.loc[df.FLT=="g",'FLT'] = 1
        # rename columns to suit preprocess_data_utils
        df = df.rename(columns = {"FIELD":"id","MJD":"time","FLT":"band"})
        
        df_tag = df_tags(df, tags[model])

        # print(df_tag)
        # print(df.head())

        # print("shape of df ", df.groupby("id").count().shape[0])
        # print("shape of tags ",df_tag.shape)
        X,id,Y = create_interpolated_vectors(df,df_tag,128,n_channels=4)
        int_ids = np.array(list(map(int, id))) 
        # print("shape of vectors", X.shape)
        # print("shape of tags", Y.shape)
        c+=len(id)
        # print(c)
        if len(data) == 0:
            data = X
            labels = Y
            ids = int_ids
        else:
            data = np.concatenate((data, X))
            labels = np.concatenate((labels, Y))
            ids = np.concatenate((ids, int_ids))

        # print("shape of concat vectors", data.shape)
        # print("shape of concat tags", labels.shape)


    dataset = {
        'X':data,
        'Y':labels,
        'ids':ids
    }

    if i == 0:
        save_vectors(dataset,"rapid_data.h5")
        counts = {model:c}
        print(counts)

    else :
        append_vectors(dataset,"rapid_data.h5")
        counts[model]=c
        print(counts)

# header_filname = "../../data/rapid_data/ZTF_20190512/ZTF_MSIP_MODEL01/ZTF_MSIP_NONIaMODEL0-0001_HEAD.FITS"
# def plot_light_curve():
#     dat = Table.read(filename, format='fits')
#     df = dat.to_pandas()
#     fields = df.FIELD.unique()
#     print(fields.shape)
#     for i in fields[10:100]:
#         id_0 = i
#         print(id_0)
#         obj_0 = df[df["FIELD"]==id_0]
#         bands = obj_0["FLT"].unique()
#         print(bands)
#         lc_0 = obj_0[obj_0["FLT"]==bands[0]].sort_values(["MJD"])
#         lc_1 = obj_0[obj_0["FLT"]==bands[1]].sort_values(["MJD"])
#         # print(lc_0)
#         # print(lc_1)
#         plt.plot(lc_0["MJD"], lc_0["FLUXCAL"],color="red",marker="o")
#         plt.plot(lc_1["MJD"], lc_1["FLUXCAL"],color="green",marker="o")
#         plt.show()

# plot_light_curve()