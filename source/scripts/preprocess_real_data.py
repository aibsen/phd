from distutils.command.clean import clean
from fileinput import filename
from tokenize import group
from venv import create
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess_data_utils
import matplotlib.pyplot as plt


# (r=0, g=1)

data_dir = "../../data/ztf/csv/tns/"

ztf_type_dict = {
    '0':'SN-Ia',
    '1':'SN-Ia-91BG', 
    '2':'SN-Iax',  
    '3':'SN-II', 
    '4':'SN-Ib/c',
    '5':'SLSN',
}

ztf_names = [ztf_type_dict[k] for k in ztf_type_dict]


def filter_sn_only(fn):
    df = pd.read_csv(data_dir+fn)   
    print(df.predicted_class.unique())
    sn_only = df[df.predicted_class.isin(['SNII','SNIbc','SNIa','SLSN'])]
    sn_only.to_csv(data_dir+"alerce_sn_probs.csv", index=False)

def compare():


    print(len(df_lasair.objectId.unique()))
    print(len(df_mars.objectId.unique()))
    # df_tags = pd.read_csv(fn_tags)
    # print(len(df_tags.objectId.unique()))

    # alerce_sn_clean = df_


    random_id = df_lasair.sample(n=1).objectId.values[0]
    print(random_id)

    obj_lasair = df_lasair[df_lasair.objectId == "ZTF19acftbsn"]
    obj_lasair = obj_lasair[obj_lasair.detected=="1"]
    obj_lasair_t_r = obj_lasair[obj_lasair["filter"] == "0"].time.values.astype("float")
    obj_lasair_r = obj_lasair[obj_lasair["filter"] =="0"].magpsf.values.astype("float")
    obj_lasair_t_g = obj_lasair[obj_lasair["filter"] =="1"].time.values.astype("float")
    obj_lasair_g = obj_lasair[obj_lasair["filter"] =="1"].magpsf.values.astype("float")


    obj_mars = df_mars[df_mars.objectId == "ZTF19acftbsn"]
    obj_mars_t_r = obj_mars[obj_mars["filter"] =="r"].time.values.astype("float")
    obj_mars_r = obj_mars[obj_mars["filter"] =="r"].magpsf.values.astype("float")
    obj_mars_t_g = obj_mars[obj_mars["filter"] =="g"].time.values.astype("float")
    obj_mars_g = obj_mars[obj_mars["filter"] =="g"].magpsf.values.astype("float")



    fig,ax = plt.subplots(2,1)

    print(obj_lasair_t_g)
    print(obj_lasair_g)

    ax[0].scatter(obj_lasair_t_r, obj_lasair_r, color='r')
    ax[0].scatter(obj_lasair_t_g, obj_lasair_g, color='g')
    ax[0].set_title('lasair')

    ax[1].scatter(obj_mars_t_r, obj_mars_r, color='r')
    ax[1].scatter(obj_mars_t_g, obj_mars_g, color='g')
    ax[1].set_title('mars')

    ax[0].set_ylim(15,22)
    ax[1].set_ylim(15,22)

    ax[0].invert_yaxis()
    ax[1].invert_yaxis()

    plt.show()


def merge_lcs():

    mars_dir = data_dir+'from_lasair/'

    lcs = []
    file_names = ['alerce_sn_usable_lc.csv','slsn-1_lc.csv','sn1a_lc.csv','sn1a-91bg_lc.csv',
    'sn1ax_lc.csv','sn1b-c_lc.csv','sn2_lc.csv']

    for file in file_names:
        df = pd.read_csv(mars_dir+file)
        df = df[df.objectId!='objectId']
        lcs.append(df)

    all_lcs = pd.concat(lcs, axis=0, ignore_index=True)
    print(all_lcs)
    all_lcs.to_csv(data_dir+"lasair_sn_lcs.csv",index=False)


def enough_points(lcs,meta,output,output_meta,required_n_points=3):
    print(lcs.objectId.unique().shape)
    enough_points = lcs.groupby(["objectId","passband"]).count()
    ids = [k for k in enough_points.index]
    print(enough_points)
    print(enough_points.describe())
    enough_points= enough_points[enough_points.mjd>=required_n_points]
    print(enough_points)
    # print(enough_points.groupby('objectId').count())
    two_bands = enough_points.groupby('objectId').count()
    # print(two_bands)
    two_bands = two_bands[two_bands.mjd==2]
    # print(two_bands)
    keys = [k for k in two_bands.index]
    # print(len(keys))
    # enough_points = enough_points
    # print(ids)
    lcs = lcs[lcs.objectId.isin(keys)]
    meta = meta[meta.objectId.isin(keys)]
    print(lcs.objectId.unique().shape)
    print(meta.objectId.unique().shape)
    lcs.to_csv(output,index=False)
    meta.to_csv(output_meta,index=False)

def clean_metadata(lcs,meta,output):
    print(lcs.objectId.unique().shape)
    print(meta.shape)
    meta = meta[meta.objectId.isin(lcs.objectId.unique())]
    # print(meta.shape)
    print(meta.objectId.unique().shape)
    # meta_dup = meta[meta.objectId.duplicated() ==True]
    # print(meta_dup)
    meta = meta.drop_duplicates(keep="first",subset="objectId")
    meta["object_id"] = np.arange(1,meta.shape[0]+1)
    print(meta)
    meta.to_csv(output)

def add_numeric_id_to_data(data,meta,output):

    data_ids = data.objectId
    meta_ids = meta[['objectId','object_id']]
    data["object_id"]=0
    for i, d in enumerate(data_ids):
        print(d)
        n=meta_ids[meta_ids.objectId == d].object_id.values[0]
        print(n)
        data.loc[i,"object_id"] = n
    data.to_csv(output,index=False)


def correct_bands():
    lcs = data_dir+"sn_lcs_drop_duplicates_enough_points_10.csv"

    data = pd.read_csv(lcs)
    print(data[data.objectId=="ZTF18aabssth"])

    def f (a):
        if a == "r" or a =="0":
          return 0
        elif a =="g" or a=="1":
            return 1
        else:
            return a

    data["passband"] = data["passband"].apply(f)
    print(data[data.objectId=="ZTF18aabssth"])
    a = data[data.objectId=="ZTF18aabssth"]
    b =a.groupby("passband").count()
    print(b)
    data.to_csv(data_dir+"sn_lcs_drop_duplicates_enough_points_10_correct_band.csv",index=False)

# correct_bands()
def add_numeric_class_to_metadata(meta,output):

    tags = [ztf_names.index(t) for t in meta.predicted_class]
    meta["true_target"] = tags
    meta.to_csv(output,index=False)


# add_numeric_class_to_metadata()
def create_interpolated_vectors(lcs,meta,output):

    X, ids = preprocess_data_utils.create_interpolated_vectors(lcs,128,n_channels=2)
    tags = meta[meta.object_id.isin(ids)].sort_values(['object_id'])
    print(tags)
    Y = tags.true_target.values

    Y = meta[meta.object_id.isin(ids)].true_target.values.astype("long")
    print(Y.shape)
    print(X.shape)
    print(ids.shape)

    print((tags['object_id'].values == ids).all())
    print((tags['object_id'].values == ids))

    print(X.shape)
    print(ids.shape)
    print(Y.shape)
    assert((tags['object_id'].values == ids).all())    

    dataset = {
        'X':X,
        'Y':Y,
        'ids':ids
    }

    preprocess_data_utils.save_vectors(dataset,output)


def merge_lasair_mars(): #merge lasair lightcurves with mars lightcurves, prioritizing lasair
    lcs = pd.read_csv(data_dir+'lasair_sn_lcs.csv')
    lasair_ids = lcs.objectId.unique()
    print(lasair_ids.shape)

    mars_lcs = pd.read_csv(data_dir+'mars_sn_lcs_clean.csv')[['objectId','mjd','passband','magpsf']]
    filtered_mars_lcs = mars_lcs[~mars_lcs.objectId.isin(lasair_ids)]
    print(filtered_mars_lcs.objectId.unique().shape)
    all_lcs = pd.concat([lcs,filtered_mars_lcs], axis=0, ignore_index=True)
    print(all_lcs)
    print(all_lcs.objectId.unique().shape)

    all_lcs.to_csv(data_dir+"lasair_mars_lcs.csv", index=False)



def at_least_a_month():
    lcs = pd.read_csv(data_dir+'mars_sn_lcs_clean.csv')
    meta = pd.read_csv(data_dir+'mars_sn_metadata_clean.csv')

    group_by_id = lcs.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'mjd_' + x).reset_index()
    group_by_id["mjd_diff"]=group_by_id.mjd_max-group_by_id.mjd_min

    print(group_by_id)
    group_by_id = group_by_id[(group_by_id['mjd_diff']>=120)]
    # &(group_by_id['mjd_diff']<=100)]
    print(group_by_id)
    ids = group_by_id.object_id
    lcs = lcs[lcs.object_id.isin(ids)]
    meta = meta[meta.object_id.isin(ids)]

    lcs.to_csv(data_dir+'mars_sn_lcs_clean_120days.csv',index=False)
    meta.to_csv(data_dir+'mars_sn_meta_clean_120days.csv',index=False)
# max_mjd = lcs[lcs]

# at_least_a_month()


def old_vectors():
    lcs = pd.read_csv(data_dir+'mars_sn_lcs_clean.csv')
    meta = pd.read_csv(data_dir+'mars_sn_metadata_clean.csv')

    group_by_id_bright = lcs.groupby(['object_id'])['magpsf'].agg(['min']).rename(columns = lambda x : 'max_magpsf').reset_index()
    merged = pd.merge(lcs,group_by_id_bright,how='left', on='object_id')
    a= merged[merged.magpsf==merged.max_magpsf][['object_id','mjd']]
    merged = pd.merge(merged,a,how='left', on='object_id',suffixes=('','_max'))

    merged = merged[np.abs(merged.mjd_max-merged.mjd)<100]
    group_by_id = merged.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'mjd_' + x).reset_index()
    group_by_id["mjd_diff"]=group_by_id.mjd_max-group_by_id.mjd_min

    long = group_by_id[group_by_id.mjd_diff>550]
    print(group_by_id.describe())

    # simlcs = pd.read_csv('/home/ai/phd/data/ztf/csv/simsurvey/simsurvey_lcs_balanced_1.csv')
    # sim_group_by_id = simlcs.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'mjd_' + x).reset_index()
    # sim_group_by_id["mjd_diff"]=sim_group_by_id.mjd_max-sim_group_by_id.mjd_min
    # print(sim_group_by_id.describe())

    # print(long.shape)

    ids_enough_obs=group_by_id[(group_by_id.mjd_diff>=90)].object_id
    group_by_id = group_by_id[group_by_id.object_id.isin(ids_enough_obs)]
    print(group_by_id)
    sn_enough=merged[merged.object_id.isin(ids_enough_obs)]

    #ensure there is at least 3 points per band
    group_by_id_band = sn_enough.groupby(['object_id','passband'])['mjd'].agg(['count']).rename(columns = lambda x : 'time_' + x).reset_index()
    #drop all lcs that have less than 3 points
    at_least_3 = group_by_id_band[group_by_id_band.time_count>=3]
    #drop also their companion band
    band_counts = at_least_3.groupby(['object_id']).count().reset_index()
    band_counts = band_counts[band_counts.time_count==2]
    sn_enough=sn_enough[sn_enough.object_id.isin(band_counts.object_id)]
    tags_enough = meta[meta.object_id.isin(band_counts.object_id)]
    print(sn_enough.object_id.unique().shape)
    print(tags_enough.shape)
    group_by_id = group_by_id[group_by_id.object_id.isin(band_counts.object_id)]
    print(group_by_id)
    print(group_by_id.describe())

    X=np.zeros((tags_enough.shape[0],4,128))
    lens=np.zeros((tags_enough.shape[0],))

    obids = tags_enough.object_id.unique()
    for n,object_id in enumerate(obids):
        lc = sn_enough[sn_enough.object_id == object_id]
        lc_r = lc[lc.passband == 0] 
        lc_g = lc[lc.passband == 1]
        print(object_id)
        lc_length = group_by_id.loc[group_by_id.object_id == object_id, 'mjd_diff'].values[0]
        lc_start = group_by_id.loc[group_by_id.object_id == object_id, 'mjd_min'].values[0]
        lc_stop = group_by_id.loc[group_by_id.object_id == object_id, 'mjd_max'].values[0]
        print(lc_length)
        print(lc_start)
        print(lc_stop)
        scaled_lc_length=int(np.floor(128*lc_length/128))
        if scaled_lc_length>=128:
            lc_step = lc_length/128
            new_x = np.arange(lc_start,lc_stop,lc_step)
            print(new_x.shape)
        else:
            lc_step = lc_length/scaled_lc_length
            new_x = np.arange(lc_start,lc_stop+1,lc_step)
            print(new_x.shape)

        X[n,0,0:new_x.shape[0]] = np.interp(new_x,lc_r.mjd, lc_r.magpsf)
        X[n,1,0:new_x.shape[0]] = np.interp(new_x,lc_g.mjd, lc_g.magpsf)
        lens[n]=new_x.shape[0]
        for i in range(new_x.shape[0]):
            X[n,2,i] = np.abs(lc_r.mjd.values - new_x[i]).min()
            X[n,3,i] = np.abs(lc_g.mjd.values - new_x[i]).min()
            
    print(X.shape)
    print(lens)
    Y = tags_enough.true_target.values
    ids = tags_enough.object_id.unique()
    dataset = {
    'X':X,
    'Y':Y,
    'ids':ids,
    'lens':lens
    }

    print(dataset)

    # X, ids = preprocess_data_utils.create_interpolated_vectors(lcs,128,n_channels=2)
    # tags = meta[meta.object_id.isin(ids)].sort_values(['object_id'])
    # print(tags)
    # Y = tags.true_target.values

    # Y = meta[meta.object_id.isin(ids)].true_target.values.astype("long")
    print(Y.shape)
    print(X.shape)
    print(ids.shape)

    print((tags_enough['object_id'].values == ids).all())
    print((tags_enough['object_id'].values == ids))

    # print(X.shape)
    # print(ids.shape)
    # print(Y.shape)
    assert((tags_enough['object_id'].values == ids).all())    

    preprocess_data_utils.save_vectors(dataset,"real_data_careful_90days.h5")
# 
    # preprocess_data_utils.save_vectors(dataset,filename)

old_vectors()

# print(sim_group_by_id.describe())

# create_interpolated_vectors(lcs,meta,'/home/ai/phd/data/ztf/testing/real_data_120days.h5')
# enough_points(lcs,meta,data_dir+'lasair_mars_lcs_10pb.csv',data_dir+'lasair_mars_meta_10pb.csv',10)

# z = meta[~meta.Redshift.isna()]
# # print(z)

# results = pd.read_csv('test_results.csv')
# results = results[results.object_id.isin(z.object_id)]
# results = results[results.target.isin([0,3,5])]
# bad=results[results.prediction!=results.target]
# good=results[results.prediction==results.target]
# print(good.shape)
# print(bad.shape)
# ids = bad.object_id.unique()


# for id in ids:
#     rid = meta[meta.object_id==id].objectId

#     g = bad[bad.object_id==id]
    
#     lc = lcs[lcs.object_id==id]
#     lc_r = lc[lc.passband==0]
#     lc_g = lc[lc.passband==1]
    
#     scatter=plt.scatter(lc_g.mjd,lc_g.magpsf,color='g')
#     plt.scatter(lc_r.mjd,lc_r.magpsf,color='r')
#     ax = scatter.axes
#     ax.invert_yaxis()
    
#     plt.title("id:{},target: {}, prediction: {}".format(rid,g.target.values,g.prediction.values))
    
#     plt.show()


# def enough_points_before_peak(lcs,meta,output,output_meta,required_n_points=3):
#     print(lcs.objectId.unique().shape)
#     enough_points = lcs.groupby(["objectId","passband"]).count()
#     ids = [k for k in enough_points.index]
#     print(enough_points)
#     print(enough_points.describe())
#     enough_points= enough_points[enough_points.mjd>=required_n_points]
#     print(enough_points)
#     # print(enough_points.groupby('objectId').count())
#     two_bands = enough_points.groupby('objectId').count()
#     # print(two_bands)
#     two_bands = two_bands[two_bands.mjd==2]
#     # print(two_bands)
#     keys = [k for k in two_bands.index]
#     # print(len(keys))
#     # enough_points = enough_points
#     # print(ids)
#     lcs = lcs[lcs.objectId.isin(keys)]
#     meta = meta[meta.objectId.isin(keys)]
#     print(lcs.objectId.unique().shape)
#     print(meta.objectId.unique().shape)
#     lcs.to_csv(output,index=False)
#     meta.to_csv(output_meta,index=False)















