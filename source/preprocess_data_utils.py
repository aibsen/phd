
import pandas as pd
import numpy as np
import h5py

# plasticc_sn_tags = {'90':0, '67':1, '52':2, '42':3, '62': 4, '95': 5}
# plasticc_sn_tags =[90,67,52,42,62,95]

"""for files in the Transient Name Server format (.csv)"""
def filter_metadata_by_type(metadata_file,types_I_want):
    """Receives a metadata file from TNS and changes tags to numbers     
    Parameters
    ----------
    metadata_file : str, path to metadata downloaded from the transient name server in .csv format
    types_I_want : dict, mapping a TNS server tag to the numerical type I want to assign

    Returns
    -------
    metadata : pandas DataFrame with numerical types instead of TNS string types
    """
    metadata = pd.read_csv(metadata_file)
    metadata = metadata[metadata["Obj. Type"].isin(types_I_want.keys())]
    for k,v in types_I_want.items():
        metadata.loc[metadata["Obj. Type"]==k,"tag"] = v
    return metadata

def merge_metadata(current_real_data_dir, n_files=5):
    """Function for reading metadata files downloaded from TNS and merging them 
    together into one for ease of use. It write the final metadata file to current_real_data_dir
    Parameters
    ----------
    current_real_data_dir: str, path to directory where file dowloaded from TNS (named tns_search.csv) is.
    n_files: int, number of total metadatdata files in the directory"""

    metadata_file = "tns_search.csv"
    df=pd.read_csv(current_real_data_dir+metadata_file)

    for i in np.arange(1,n_files+1):
        metadata_file = current_real_data_dir+"tns_search({}).csv".format(i)
        df2 = pd.read_csv(metadata_file)
        df = pd.concat([df,df2])
    df=df.drop_duplicates(keep="first")
    df.to_csv(current_real_data_dir+"tns_search_sn_metadata.csv",index=False)

"""for files in the PLAsTiCC format (DataFrame)"""

def retag_plasticc(metadata,plasticc_tags):
    """Receives a metadata DataFrame read in the PLASTiCC format and changes tags to be between 0 and the number of types wanted.    
    Parameters
    ----------
    metadata : pandas DataFrame, metadata DF read from PLASTiCC dataset
    plasticc_tags: array, contains all the PLAsTiCC types I want to use, in order
    -------
    sn_metadata : pandas DataFrame with different numerical types
    """
    #choosing given tags only
    sn_metadata = metadata[metadata["true_target"].isin(plasticc_tags)].copy()
    sn_metadata.loc[:,"true_target"] = [plasticc_tags.index(tag) for tag in sn_metadata["true_target"]]
    return sn_metadata

"""for files in Simsurvey format (.pkl)"""

def pkl_to_df(pkl_filename, first_id = 0):
    """Receives a path to a .pkl produced by simsurvey and returns a pandas DataFrame with the
    same light curves in an easier to handle format
    ----------
    pkl_filename : str, path to .pkl produced by simsurvey that contains simulated light curves
    first_id: int, optional. the first numerical id to be used when assigning it to a simulated light curve.
    -------
    df_sn : pandas DataFrame, contains input lightcurves where each row contains 
        id, time, band, flux, fluxerr for a given point in a light curve
    """
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

def is_flux_to_abmag_working(filename, trial_size):
    """Sanity check to see if flux-mag conversion is working
    Parameters
    ----------
    filaname: .pkl filename as given by simsurvey
    trial_size: the number of light curves to be checked"""
    trial = pd.read_pickle(filename)
    lcs_trial = trial["lcs"][0:trial_size]
    stats = trial["stats"]
    maxmg_trial=stats["mag_max"]["ztfg"][0:trial_size]
    maxmr_trial=stats["mag_max"]["ztfr"][0:trial_size]
    maxfg_trial=np.zeros(trial_size)
    maxfr_trial=np.zeros(trial_size)
    for count in np.arange(trial_size):
        r=np.array(list(filter(lambda p: p["band"]=='ztfr' , lcs_trial[count])))
        g=np.array(list(filter(lambda p: p["band"]=='ztfg' , lcs_trial[count])))

        maxfr_trial[count] = np.amax(np.array(list(map(lambda p: p["flux"], r))))
        maxfg_trial[count]= np.amax(np.array(list(map(lambda p: p["flux"], g))))

    df_trial = pd.DataFrame({"mg":maxmg_trial, "mr":maxmr_trial, "fg":maxfg_trial, "fr":maxfr_trial})
    df_trial["fg"]=flux_to_abmag(df_trial["fg"].values)
    df_trial["fr"]=flux_to_abmag(df_trial["fr"].values)
    if (df_trial["fg"]==df_trial["mg"]).all() and (df_trial["fr"]==df_trial["mr"]).all():
        print("yes, it is")
    else:
        print("no, it ain't")    

"""Miscellaneous"""
def df_tags(df_sn, t):
    """Receives a Dataframe with lightcurves of a single type 
    and returns column with types for all unique ids
    Parameters
    ----------
    df_sn : pandas DataFrame, data DF in PLASTiCC format
    t: int, tag of type
    Returns
    -------
    df_sn_tags : pandas column of types for unique ids
    """
    sn_ids = df_sn.id.unique()
    df_sn_tags = pd.DataFrame(data=sn_ids, columns = ["id"])
    df_sn_tags.loc[:,"type"] = t
    return df_sn_tags

def flux_to_abmag(f,zp=30):
    """Converts a flux to ab magnitude
    Parameters
    ----------
    f: float, flux
    zp: float, zero point value
    Returns
    -------
    ab magnitude
    """
    return zp-2.5*np.log10(f)

def abmag_to_flux(mag,zp=30):
    """Converts ab_magnitude to flux
    Parameters
    ----------
    mag: magnitude
    zp: float, zero point value
    Returns
    -------
    flux
    """
    return np.power(10,(zp-mag)/2.5)

""" Lasair utilities """
def load_real_lcs(sn_filename):
    """Takes a .csv file in Lasair format, drops nans, changes column names and 
    bands to make them consisten with simsurvey data
    ----------
    sn_filename: str, path to .csv file in Lasair format
    Returns
    -------
    Pandas DataFrame with updated data
    """
    sn = pd.read_csv(sn_filename,sep="|").dropna(axis=1)
    #rename columns
    sn.columns = ["id","time","flux","flux_err","band"]
    #make passbands consistent with simulated data (0,1 instad of 1,2)
    sn.loc[sn["band"]==2,"band"] = 0
    return sn

def ids_for_lasair(metadata_file):
    """it takes a tns metafile and returns a string-like list of ids to use to query lasair with
    Parameters
    ----------
    metadat_file: str,path to tns like .csv containing metadat
    Returns
    -------
    ids_str: str with all ids to get from lasair
    """
    real_sns = pd.read_csv(metadata_file)
    ids = real_sns["Disc. Internal Name"].drop_duplicates()
    ids = ids.dropna()
    ids = ids[ids.str.contains("ZTF")]
    ids = ids.str.split(",").apply(lambda x: x[0].strip() if "ZTF" in x[0] else x[1].strip()).values
    id_str = ''
    for i in ids:
        id_str= id_str+'"'+i+'", '#fix this
    return id_str

"""Functions to generate .hdf5 files that will be loaded as datasets"""

def create_interpolated_vectors(data, tags, length=128, n_passbands=2):
    """Takes data in the PLAsTiCC format and the corresponding tags and returns the data
    in the form of linearly interpolated vectors of size length, with extra channels that meassure
    distances between interpolations and nearest real points. Dismisses fluxerrors.
    This algorithm was originally coded by mammas for the PLAsTiCC challenge
    Parameters
    ----------
    data: pandas DataFrame, contains data where each row is a lightcurve point.
        PLAsTiCC columns are is object_id,mjd,flux, passband,
        Simsurvey format is id, time, flux, band
        #need to fix this
    tags: pandas DataFrame, single column with tags of objects ordered by id. 
        PLAsTiCC column is true_target
        Simsurvey column is type
        #need to fix this
    length: int, optional. Desired length of interpolated light curves.
    n_passbands: int, optional. Number of passbands in an object
        PLAsTicc objects have 6 passbands
        Simsurvey objects have 2 passbands
    Returns
    -------
    (X, ids, Y) : array with interpolated vectors, ids and tags for them
    """
    obj_ids = tags.id.unique()
    data_cp = data.copy()
    if data.band:#then format is simsurvey like
        data_cp['ob_p']=data.id+data.band.apply(lambda band: str(band))
    elif data.passband:#then format is plasticc like and we need to change it
        data_cp['ob_p']=data.object_id*10+data.passband
        data_cp=data_cp.rename(columns={"object_id": "id", "mjd": "time","passband":"band"})
        tags = tags.rename(columns={"true_target":"type"})
    #sanity check
    # print("there are",data_cp.id.unique().size, "objects")
    # print("there are",data_cp.ob_p.unique().size, "lightcurves")
    # print("is the n_lcs = n_passbands times n_objs?",data_cp.id.unique().size*n_passbands==data_cp.ob_p.unique().size)
    
    #get dataframe with min and max mjd values per each object id
    group_by_mjd = data_cp.groupby(['id'])['time'].agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    merged = pd.merge(data_cp, group_by_mjd, how = 'left', on = 'id')
    #sanity check
    # print("do I still have the same nobjs",merged.id.unique().size == data_cp.id.unique().size)

    #scale mjd according to max mjd, min mjd and the desired length of the light curve (128 default)
    #here I need to think wether I really want the time scaled
    merged['scaled_time'] = (length - 1) * (merged['time'] - merged['time_min'])/(merged['time_max']-merged['time_min'])
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    merged=merged.sort_values(['id','time'])
    #sanity check
    # print("still the same number of objects?",merged.id.unique().size==data_cp.id.unique().size)

    #reshape df so that for each row there's one lightcurve (n_passbands rows per obj) and each column is a point of it
    # there is two main columns also, for flux and for mjd
    unstack = merged[['ob_p', 'scaled_time', 'flux', 'cc']].set_index(['ob_p', 'cc']).unstack()
    # print("still same number of objects when unstacking?",unstack.shape[0]== data_cp.id.unique().size*n_passbands)
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
    for i in range(n_lcs):
        if nan_masks[i].any(): #if any point is real
            X[i] = np.interp(x, time_flux[i][:, 0][nan_masks[i]], time_flux[i][:, 1][nan_masks[i]])
        else:
            X[i] = np.zeros_like(x)

    n_objs = int(n_lcs/n_passbands)
    #reshape vectors so the ones belonging to the same object are grouped into 2 channels
    X_per_band = X.reshape((n_objs,n_passbands,length)).astype(np.float32)
    #get distance for each point to nearest real point
    X_void = np.zeros((n_lcs, x.shape[0]))
    for i in range(length):
        X_void[:, i] = np.abs((unstack["scaled_time"] - i)).min(axis = 1).fillna(500)
    #reshape vectors so the ones belonging to the same object are grouped into n_passbands channels
    X_void_per_band = X_void.reshape((n_objs,n_passbands,length)).astype(np.float32)
    vectors = np.concatenate((X_per_band,X_void_per_band),axis=1)
    return vectors, obj_ids, tags.type.values

"""Functions to save and update .hdf5 generated files"""
def append_vectors(dataset,outputFile):
    """It appends generated dataset dictionary into an existing .hdf5 file
    Parameters
    ----------
    dataset: dict, dataset in the format {"X":,"ids":,"Y":}
    outputFile: str, path to .hdf5 file to update
    """
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
    """It wrotes generated dataset dictionary into a new .hdf5 file
    Parameters
    ----------
    dataset: dict, dataset in the format {"X":,"ids":,"Y":}
    outputFile: str, path to .hdf5 ouput
    """
    hf=h5py.File(outputFile,'w')

    print("writing X")
    hf.create_dataset('X',data=dataset['X'],compression="gzip", chunks=True, maxshape=(None,None,None,))

    print("writing ids")
    hf.create_dataset('ids',data=dataset['ids'],dtype='int64',compression="gzip", chunks=True, maxshape=(None,))
    
    print("writing Y")
    hf.create_dataset('Y',data=dataset['Y'],compression="gzip", chunks=True, maxshape=(None,))
    hf.close()