
import pandas as pd
import numpy as np
import h5py

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
def df_tags(df_sn, type):
    sn_ids = df_sn.id.unique()
    df_sn_tags = pd.DataFrame(data=sn_ids, columns = ["id"])
    df_sn_tags.loc[:,"type"] = type
    return df_sn_tags

def create_interpolated_vectors(data, tags, length):
    obj_ids = tags.id.unique()
    #different ids for different passbands
    data_cp = data.copy()
    data_cp['ob_p']=data.id*10+data.band
    #print(data_cp.id.unique().size)
    #print(data_cp.ob_p.unique().size)
    #print(data_cp.id.unique().size*2==data_cp.ob_p.unique().size)
    # rem=set(obj_ids_p).difference(set(data_cp['ob_p'].values))
    # if len(rem)>0:
    #     mmjd=data_cp.time.mean()
    #     data_rem=np.zeros((len(rem),6))
    #     rml=np.array(list(rem))
    #     data_rem[:,0] = (rml/10).astype('int')
    #     data_rem[:,1] = np.ones(len(rem))*mmjd 
    #     data_rem[:,2]= (rml-data_rem[:,0]*10).astype('int')
    #     data_rem[:,5]=rml
    #     df_rem=pd.DataFrame(data=data_rem, columns=['id','time','band','flux','fluxerr','ob_p'])
    #     data_cp=pd.concat([data_cp,df_rem],ignore_index=True).sort_values(['id','time']).reset_index(drop=True)
       
    #get dataframe with min and max mjd values per each object id
    group_by_mjd = data_cp.groupby(['id'])['time'].agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    merged = pd.merge(data_cp, group_by_mjd, how = 'left', on = 'id')
    #print(merged.id.unique().size == data_cp.id.unique().size)
    #scale mjd according to max mjd, min mjd and the desired length of the light curve (128)
    merged['scaled_time'] = (length - 1) * (merged['time'] - merged['time_min'])/(merged['time_max']-merged['time_min'])
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    merged=merged.sort_values(['id','time'])
    #print(merged.id.unique().size==data_cp.id.unique().size)
    #print(data_cp.id.unique().size)
    
    #reshape df so that for each row there's one lightcurve (2 rows per obj) and each column is a point of it
    # there is two main columns also, for flux and for mjd
    unstack = merged[['ob_p', 'scaled_time', 'flux', 'cc']].set_index(['ob_p', 'cc']).unstack()
    #print(unstack.shape)
    #print(unstack.shape[0]== data_cp.id.unique().size*2)
    #transform above info into numpy arrays
    time_uns = unstack['scaled_time'].values[..., np.newaxis]
    flux_uns = unstack['flux'].values[..., np.newaxis]
    time_flux = np.concatenate((time_uns, flux_uns), axis =2)
    #print(flux_uns.shape)
    #print(time_uns.shape)
    #print(time_flux.shape)
    #create a mask to get points that are valid (not nan)
    nan_masks = ~np.isnan(time_flux)[:, :, 0]
    x = np.arange(length)
    
    n_lcs = time_flux.shape[0]
    #here we'll store interpolated lcs
    X = np.zeros((n_lcs, x.shape[0]))
    X_void = np.zeros((n_lcs, x.shape[0]))
    
    t=range(n_lcs)
    #print(n_lcs)
    #here we'll store the channels that tells us how far a point is from the nearest real point
    #interpolation
    for i in t: 
        if nan_masks[i].any():
            X[i] = np.interp(x, time_flux[i][:, 0][nan_masks[i]], time_flux[i][:, 1][nan_masks[i]])
        else:
            X[i] = np.zeros_like(x)
    # print(X.shape)
    #get distance for each point to nearest real point
    t=range(length)
    for i in t:
        X_void[:, i] = np.abs((unstack["scaled_time"] - i)).min(axis = 1).fillna(500)

    #reshape vectors so the ones belonging to the same object are grouped into 2 channels  
    #print(X)
    n_objs = int(X.shape[0]/2)
    X_per_band = X.reshape((n_objs,2,length)).astype(np.float32)
    # print(X_per_band.shape)
    X_void_per_band = X_void.reshape((n_objs,2,length)).astype(np.float32)
    # print(X_void_per_band.shape)
    vectors = np.concatenate((X_per_band,X_void_per_band),axis=1)
    # print(vectors.shape)
    return vectors, obj_ids, tags.type.values

def save_vectors(dataset, outputFile):
    hf=h5py.File(outputFile,'w')
    print("writing X")
    hf.create_dataset('X',data=dataset['X'])
    print("writing ids")
    hf.create_dataset('ids',data=dataset['ids'])
    print("writing Y")
    hf.create_dataset('Y',data=dataset['Y'])
    hf.close()

def flux_to_abmag(f,zp=30):
    return 30-2.5*np.log10(f)

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