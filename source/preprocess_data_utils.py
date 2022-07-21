
import pandas as pd
import numpy as np
import h5py

# plasticc_sn_tags = {'90':0, '67':1, '52':2, '42':3, '62': 4, '95': 5}
plasticc_sn_tags =[90,67,52,42,62,95]

def retag_plasticc(metadata):
    #choosing supernovae only
    sn_metadata = metadata[metadata["true_target"].isin(plasticc_sn_tags)].copy()
    sn_metadata.loc[:,"true_target"] = [plasticc_sn_tags.index(tag) for tag in sn_metadata["true_target"]]
    return sn_metadata

def filter_metadata_by_type(metadata,types_I_want):
#receives a dict with of types by name as provided by TNS and the numerical tags I want to give them
    metadata = metadata[metadata["Obj. Type"].isin(types_I_want.keys())]
    for k,v in types_I_want.items():
        metadata.loc[metadata["Obj. Type"]==k,"tag"] = v
    return metadata

#receives a filename that contains the simsurvey simulated lightcurves and returns
# a dataframe with the light curves with a sensible format for easier handling.
def pkl_to_df(pkl_filename, first_id = 0):
    sn_data = pd.read_pickle(pkl_filename)
    df = pd.DataFrame(data=sn_data["lcs"], dtype=np.int8)
    df_sn = df.stack(dropna=True)
    df_sn = df_sn.to_frame()
    df_sn.reset_index(level=0, inplace=True)
    df_sn=df_sn.rename(columns={"level_0": "id", 0: "X"})
    df_sn["id"]= df_sn["id"]+first_id
    df_sn["time"] = df_sn["X"].str[0]
    df_sn["band"] = df_sn["X"].str[1]
    df_sn["flux"] = df_sn["X"].str[2]
    df_sn["fluxerr"] = df_sn["X"].str[3]
    df_sn= df_sn.drop("X",axis=1)
    df_sn=df_sn[df_sn.band != 'desi']
    df_sn.loc[df_sn.band == 'ztfr', 'band'] = 0
    df_sn.loc[df_sn.band == 'ztfg', 'band'] = 1
    return df_sn


#receives dataframe with lightcurves of a type
#returns a series? with a tag for all of the ids
def df_tags(df_sn, t):
    # print(len(df_sn))
    sn_ids = df_sn.id.unique()
    df_sn_tags = pd.DataFrame(data=sn_ids, columns = ["id"])
    df_sn_tags.loc[:,"type"] = t
    return df_sn_tags

def create_interpolated_vectors(data, tags, length, n_channels=2):
    obj_ids = tags.id.unique()
    data_cp = data.copy()
    # if dtype == 'sim':
    # data_cp['ob_p']=data.id*10+data.band
    # elif dtype == 'real':
    data_cp['ob_p']=data.id+data.band.apply(lambda band: str(band))

    #sanity check
    # print("there are",data_cp.id.unique().size, "objects")
    # print("there are",data_cp.ob_p.unique().size, "lightcurves")
    # print("is the n_lcs twice n_objs?",data_cp.id.unique().size*2==data_cp.ob_p.unique().size)

    #get dataframe with min and max mjd values per each object id
    group_by_mjd = data_cp.groupby(['id'])['time'].agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    merged = pd.merge(data_cp, group_by_mjd, how = 'left', on = 'id')

    #sanity check
    # print("do I still have the same nobjs",merged.id.unique().size == data_cp.id.unique().size)

    #scale mjd according to max mjd, min mjd and the desired length of the light curve (128)
    merged['scaled_time'] = (length - 1) * (merged['time'] - merged['time_min'])/(merged['time_max']-merged['time_min'])
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    merged=merged.sort_values(['id','time'])
    #sanity check
    # print("still?",merged.id.unique().size==data_cp.id.unique().size)

    #reshape df so that for each row there's one lightcurve (2 rows per obj) and each column is a point of it
    # there is two main columns also, for flux and for mjd
    unstack = merged[['ob_p', 'scaled_time', 'flux', 'cc']].set_index(['ob_p', 'cc']).unstack()
    # print("still when unstacking?",unstack.shape[0]== data_cp.id.unique().size*2)
    #transform above info into numpy arrays
    time_uns = unstack['scaled_time'].values[..., np.newaxis]
    flux_uns = unstack['flux'].values[..., np.newaxis]
    time_flux = np.concatenate((time_uns, flux_uns), axis =2)
    #create a mask to get points that are valid (not nan)
    #do this for time dim only, since fluxes will be nan when times are also
    nan_masks = ~np.isnan(time_flux)[:, :, 0]
    x = np.arange(length)
    n_lcs = time_flux.shape[0]
    #here we'll store interpolated lcs
    X = np.zeros((n_lcs, x.shape[0]))
    t=range(n_lcs)
    for i in t:
        if nan_masks[i].any(): #if any point is real
            X[i] = np.interp(x, time_flux[i][:, 0][nan_masks[i]], time_flux[i][:, 1][nan_masks[i]])
        else:
            X[i] = np.zeros_like(x)

    n_objs = int(n_lcs/2)
    #reshape vectors so the ones belonging to the same object are grouped into 2 channels
    X_per_band = X.reshape((n_objs,2,length)).astype(np.float32)

    if n_channels == 4:
    #get distance for each point to nearest real point
        X_void = np.zeros((n_lcs, x.shape[0]))
        t=range(length)
        for i in t:
            X_void[:, i] = np.abs((unstack["scaled_time"] - i)).min(axis = 1).fillna(500)

        #reshape vectors so the ones belonging to the same object are grouped into 2 channels
        X_void_per_band = X_void.reshape((n_objs,2,length)).astype(np.float32)
        vectors = np.concatenate((X_per_band,X_void_per_band),axis=1)
        return vectors, obj_ids, tags.type.values

    elif n_channels == 2:
        return X_per_band, obj_ids, tags.type.values

def create_interpolated_vectors_plasticc(data, length, n_channels=6):
    # obj_ids = tags.object_id.unique()
    data_cp = data.copy()
    data_cp['ob_p']=data.object_id*10+data.passband

    #sanity check, 6 lcs per object
    assert(data_cp.object_id.unique().size*n_channels==data_cp.ob_p.unique().size)

    #get dataframe with min and max mjd values per each object id
    group_by_mjd = data_cp.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    merged = pd.merge(data_cp, group_by_mjd, how = 'left', on = 'object_id')

    #sanity check, still same number of objects
    assert(merged.object_id.unique().size == data_cp.object_id.unique().size)

    #scale mjd according to max mjd, min mjd and the desired length of the light curve (128)
    merged['scaled_time'] = (length - 1) * (merged['mjd'] - merged['time_min'])/(merged['time_max']-merged['time_min'])
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    merged=merged.sort_values(['object_id','mjd'])
    #sanity check
    assert(merged.object_id.unique().size==data_cp.object_id.unique().size)

    #reshape df so that for each row there's one lightcurve (6 rows per obj) and each column is a point of it
    # there is two main columns also, for flux and for mjd
    unstack = merged[['ob_p', 'scaled_time', 'flux', 'cc']].set_index(['ob_p', 'cc']).unstack()
    print(merged)
    print(unstack)
    print(merged.object_id.unique())
    # sanity check
    assert(unstack.shape[0]== data_cp.object_id.unique().size*n_channels)
    #transform above info into numpy arrays
    time_uns = unstack['scaled_time'].values[..., np.newaxis]
    flux_uns = unstack['flux'].values[..., np.newaxis]
    time_flux = np.concatenate((time_uns, flux_uns), axis =2)
    #create a mask to get points that are valid (not nan)
    #do this for time dim only, since fluxes will be nan when times are also
    nan_masks = ~np.isnan(time_flux)[:, :, 0]
    x = np.arange(length)
    n_lcs = time_flux.shape[0]
    #here we'll store interpolated lcs
    X = np.zeros((n_lcs, x.shape[0]))
    t=range(n_lcs)
    for i in t:
        if nan_masks[i].any(): #if any point is real
            X[i] = np.interp(x, time_flux[i][:, 0][nan_masks[i]], time_flux[i][:, 1][nan_masks[i]])
        else:
            X[i] = np.zeros_like(x)

    n_objs = int(n_lcs/n_channels)
    #reshape vectors so the ones belonging to the same object are grouped into 6 channels
    X_per_band = X.reshape((n_objs,n_channels,length)).astype(np.float32)

    #get distance for each point to nearest real point
    X_void = np.zeros((n_lcs, x.shape[0]))
    t=range(length)
    for i in t:
        X_void[:, i] = np.abs((unstack["scaled_time"] - i)).min(axis = 1).fillna(500)

    #reshape vectors so the ones belonging to the same object are grouped into 6 channels
    X_void_per_band = X_void.reshape((n_objs,n_channels,length)).astype(np.float32)
    vectors = np.concatenate((X_per_band,X_void_per_band),axis=1)
    return vectors, merged.object_id.unique()


def append_vectors(dataset,outputFile):
    with h5py.File(outputFile, 'a') as hf:
        X=dataset["X"]
        hf["X"].resize((hf["X"].shape[0] + X.shape[0]), axis = 0)
        hf["X"][-X.shape[0]:] = X

        ids = dataset["ids"]
        hf["ids"].resize((hf["ids"].shape[0] + ids.shape[0]), axis = 0)
        hf["ids"][-ids.shape[0]:] = ids

        Y=dataset["Y"]
        hf["Y"].resize((hf["Y"].shape[0] + Y.shape[0]), axis = 0)
        hf["Y"][-Y.shape[0]:] = Y
        hf.close()



def save_vectors(dataset, outputFile):
    hf=h5py.File(outputFile,'w')

    print("writing X")
    hf.create_dataset('X',data=dataset['X'],compression="gzip", chunks=True, maxshape=(None,None,None,))

    print("writing ids")
    hf.create_dataset('ids',data=dataset['ids'],dtype='int64',compression="gzip", chunks=True, maxshape=(None,))
    
    print("writing Y")
    hf.create_dataset('Y',data=dataset['Y'],compression="gzip", chunks=True, maxshape=(None,))
    hf.close()

def flux_to_abmag(f,zp=30):
    return zp-2.5*np.log10(f)

def abmag_to_flux(mag,zp=30):
    return np.power(10,(zp-mag)/2.5)

def is_flux_to_abmag_working(filename):
    trial = pd.read_pickle(filename)
    lcs_100 = trial["lcs"][0:100]
    stats = trial["stats"]
    maxmg_100=stats["mag_max"]["ztfg"][0:100]
    maxmr_100=stats["mag_max"]["ztfr"][0:100]
    maxfg_100=np.zeros(100)
    maxfr_100=np.zeros(100)
    for count in np.arange(100):
        r=np.array(list(filter(lambda p: p["band"]=='ztfr' , lcs_100[count])))
        g=np.array(list(filter(lambda p: p["band"]=='ztfg' , lcs_100[count])))

        maxfr_100[count] = np.amax(np.array(list(map(lambda p: p["flux"], r))))
        maxfg_100[count]= np.amax(np.array(list(map(lambda p: p["flux"], g))))
    df_trial_100 = pd.DataFrame({"mg":maxmg_100, "mr":maxmr_100, "fg":maxfg_100, "fr":maxfr_100})
    df_trial_100["fg"]=flux_to_abmag(df_trial_100["fg"].values)
    df_trial_100["fr"]=flux_to_abmag(df_trial_100["fr"].values)
    if (df_trial_100["fg"]==df_trial_100["mg"]).all() and (df_trial_100["fr"]==df_trial_100["mr"]).all():
        print("yes is is")
    else:
        print("no it ain't")

def load_real_lcs(sn_filename):
    sn = pd.read_csv(sn_filename,sep="|").dropna(axis=1)
    sn.dropna(axis=1)
    #rename columns
    sn.columns = ["id","time","flux","flux_err","band"]
    #make passbands consistent with simulated data (0,1 instad of 1,2)
    sn.loc[sn["band"]==2,"band"] = 0
    return sn

def check_lc_length(data, percentile=None):
    #td_threshold: time difference threshold in mjd

    #different ids for different passbands
    data_cp = data.copy()
    # print(data_cp)
    # data_cp['id_b']=data.id+data.band.astype('str')
    data_cp['id_b']=data.id+data.band.apply(lambda band: str(band))

    #get ids of real objects that have at least d days of observations
    #d is defined as the value which percentile% of the data falls below
    group_by_id = data_cp.groupby(['id'])['time'].agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    group_by_id["time_diff"]=group_by_id.time_max-group_by_id.time_min
    group_by_id_band = data_cp.groupby(['id','band'])['time'].agg(['count']).rename(columns = lambda x : 'time_' + x).reset_index()

    if percentile :
        td_stats=group_by_id.describe()
        #get percentile threshold value. x% of data falls below this value.
        td_threshold=td_stats.loc[percentile,'time_diff']
        print("time diff threshold for percentile ",percentile, " is ",td_threshold)
        ids_enough_obs_days=group_by_id[group_by_id.time_diff>td_threshold].id.values
        #get ids of real objects that have at least c points per passband
        #c is defined as the value which percentile% of the data falls below
        # td_stats=group_by_id_band.describe()
        # tc_threshold=td_stats.loc[percentile,'time_count']
        # print("point count threshold for percentile ",percentile, " is ",tc_threshold)
        # group_by_id_band=group_by_id_band[group_by_id_band.time_count>tc_threshold]

    else :
        ids_enough_obs_days=group_by_id[group_by_id.time_diff>40].id.values
        td_threshold=40

    group_by_id_band = group_by_id_band.groupby(['id']).count()
    ids_enough_point_count = group_by_id_band[group_by_id_band.time_count==2]
    return td_threshold, list(set(ids_enough_point_count.index.values))

def ids_for_lasair():
    """it takes a tns metafile and returns a string-like list of ids to use to query lasair with
    !!!need to check out format"""
    real_sns = pd.read_csv(metadata_file)

    ids = real_sns["Disc. Internal Name"].drop_duplicates()
    ids = ids.dropna()
    ids = ids[ids.str.contains("ZTF")]
    ids = ids.str.split(",").apply(lambda x: x[0].strip() if "ZTF" in x[0] else x[1].strip()).values

    id_str = ''
    for i in ids:
        id_str= id_str+'"'+i+'", '
    
    return id_str

# data_file="tns_search_sn_metadata.csv"
# metadata_file = current_real_data_dir+data_file
# colors = ['#00dbdd','#fd8686','#f9d62d','#b5d466','#ffa77c']
# sn_dict = {'SN Ia':'Ia', 'SN Ib':'Ib/c', 'SN Ic':'Ib/c', 'SN II':'II','SLSN':'SLSN'}
# plot_sns_by_type(sn_dict, metadata_file, colors)
# plot_sns_by_date(metadata_file,colors[-1])

# def merge_metadata(current_real_data_dir, n_files=5):
# #this function is for reading metadata files downloaded from tns server and merging
# #them together into one for later easier analysis. it receives the initial data
# #directory and the number of files in it.

#     data_file = "tns_search.csv"
#     df=pd.read_csv(current_real_data_dir+data_file)

#     for i in np.arange(1,n_files+1):
#         data_file = current_real_data_dir+"tns_search({}).csv".format(i)
#         print(data_file)
#         df2 = pd.read_csv(data_file)
#         df = pd.concat([df,df2])
#         # print(df.head())
#         print(df.shape)

#     print(df.keys())
#     df=df.drop_duplicates(keep="first")

#     df.to_csv(current_real_data_dir+"tns_search_sn_metadata.csv",index=False)