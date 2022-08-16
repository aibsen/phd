import pandas as pd
import random
import numpy as np
import torch
# from utils import load_meta_data, load_csv_data, retag, extragalactic
import matplotlib.pyplot as plt
import h5py
# from preprocess_plasticc_data_args import get_args

defaultDir = '../../data/plasticc/csvs/'

extragalactic = [90,67,52,42,62,95,15,64,88]
extragalactic_names = [
    "SNIa",
    "SNIa91bg",
    "SNIax",
    "SNII",
    "SNIbc",
    "SLSN-I",
    "TDE",
    "KN",
    "AGN"]

# def get_class_balance():
#     m = load_meta_data(defaultDir+"plasticc_metadata.csv")
#     ddf = m[m["ddf_bool"] ==1]
#     wdf = m[m["ddf_bool"] ==0] #select WDF survey strategy only
#     objs_per_class = m.groupby("true_target").count()["object_id"].rename("objs_per_class")
#     weights = (objs_per_class/m.shape[0]).rename("weights")

#     objs_per_class_wdf = wdf.groupby("true_target").count()["object_id"].rename("objs_per_class_wdf")
#     weights_wdf = (objs_per_class/wdf.shape[0]).rename("weights_wdf")

#     objs_per_class_ddf = ddf.groupby("true_target").count()["object_id"].rename("objs_per_class_ddf")
#     weights_ddf = (objs_per_class/ddf.shape[0]).rename("weights_ddf")

#     class_balance=pd.concat([objs_per_class,objs_per_class_ddf,objs_per_class_wdf, weights,weights_ddf,weights_wdf],axis=1).reset_index()

#     class_balance.to_csv("class_balance.csv", index=False)
                
def save_vectors(dataset, outputFile):
    hf=h5py.File(outputFile,'w')
    print("writing X")
    hf.create_dataset('X',data=dataset['X'])
    print("writing ids")
    hf.create_dataset('ids',data=dataset['ids'])
    print("writing Y")
    hf.create_dataset('Y',data=dataset['Y'])
    hf.close()

def retag(targets):
    retagged = targets
    for i,c in enumerate(extragalactic):
        retagged.loc[retagged==c] = i
    return retagged


def load_csv_data(data_file, metadata_file):
        data = pd.read_csv(data_file, sep=",", names=["object_id","mjd","passband","flux","flux_err","detected_bool"], skiprows=1)
        ids = data['object_id']
        ids = ids.drop_duplicates().values
        metadata = pd.read_csv(metadata_file, sep=",", names=["object_id","ra","decl",
        "ddf_bool","hostgal_specz","hostgal_photoz","hostgal_photoz_err","distmod","mwebv",
        "target","true_target","true_submodel","true_z","true_distmod","true_lensdmu",
        "true_vpec","true_rv","true_av","true_peakmjd","libid_cadence","tflux_u","tflux_g",
        "tflux_r","tflux_i","tflux_z","tflux_y"], skiprows=1)
        return [data, ids, metadata]

# # fast interpulation for a full dataset of timeseries
# # The Original fast algorithm was designed by @Mamas (whose github I can no longer find for some reason)
def create_interpolated_vectors(data_csv, metadata_csv,output_dir, length):
    #load data
    data, ids, metadata = load_csv_data(data_csv, metadata_csv)

    data_cp = data.copy()
    #get ids
    obj_ids = data.object_id.unique()
    #get targets and retag them (so the classes are numbered from 0 to 14)
    m =  metadata.loc[metadata["object_id"].isin(obj_ids)].drop_duplicates("object_id")
    targets = m["true_target"]
    mask = targets > 100 #class 99 mask
    targets.loc[mask] = 99
    classes_ = targets.drop_duplicates().values
    new_targets = retag(targets)

    #add a number from 0 to 5 at the end of the id so there is an id per passband
    obj_ids_p=np.concatenate([10*obj_ids + d for d in range(6)])
    data_cp['ob_p']=data.object_id*10+data.passband
    
    rem=set(obj_ids_p).difference(set(data_cp['ob_p'].values))

    if len(rem)>0:
        mmjd=data_cp.mjd.mean()
        data_rem=np.zeros((len(rem),7))
        rml=np.array(list(rem))
        data_rem[:,0] = (rml/10).astype('int')
        data_rem[:,1] = np.ones(len(rem))*mmjd 
        data_rem[:,2]= (rml-data_rem[:,0]*10).astype('int')
        data_rem[:,6]=rml
        df_rem=pd.DataFrame(data=data_rem, columns=['object_id','mjd','passband','flux','flux_err','detected','ob_p'])
        data_cp=pd.concat([data_cp,df_rem],ignore_index=True).sort_values(['object_id','mjd']).reset_index(drop=True)

    #catch above rem problem later

    #get dataframe with min and max mjd values per each object id
    group_by_mjd = data_cp.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'mjd_' + x).reset_index()
    #add this info to data
    merged = pd.merge(data_cp, group_by_mjd, how = 'left', on = 'object_id')
    #scale mjd according to max mjd, min mjd and the desired length of the light curve (128)
    merged['mm_scaled_mjd'] = (length - 1) * (merged['mjd'] - merged['mjd_min'])/(merged['mjd_max']-merged['mjd_min'])
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    merged=merged.sort_values(['object_id','mjd'])
    #reshape df so that for each row there's one lightcurve (6 rows per obj) and each column is a point of it
    # there is two main columns also, for flux and for mjd
    unstack = merged[['ob_p', 'mm_scaled_mjd', 'flux', 'cc']].set_index(['ob_p', 'cc']).unstack()
    #transform above info into numpy arrays
    mjd_uns = unstack['mm_scaled_mjd'].values[..., np.newaxis]
    flux_uns = unstack['flux'].values[..., np.newaxis]
    mjd_flux = np.concatenate((mjd_uns, flux_uns), axis =2)
    #create a mask to get points that are valid (not nan)
    nan_masks = ~np.isnan(mjd_flux)[:, :, 0]
    x = np.arange(length)
    
    #here we'll store interpolated lcs
    X = np.zeros((mjd_flux.shape[0], x.shape[0]))
    t=range(mjd_flux.shape[0])
    #here we'll store the channels that tells us how far a point is from the nearest real point
    X_void = np.zeros((unstack.shape[0], x.shape[0]))
    
    #interpolation
    for i in t: 
        if nan_masks[i].any():
            X[i] = np.interp(x, mjd_flux[i][:, 0][nan_masks[i]], mjd_flux[i][:, 1][nan_masks[i]])
        else:
            X[i] = np.zeros_like(x)
    #get distance for each point to nearest real point
    t=range(length)
    for i in t:
        X_void[:, i] = np.abs((unstack["mm_scaled_mjd"] - i)).min(axis = 1).fillna(500)

    #reshape vectors so the ones belonging to the same object are grouped into 6 channels    
    n_objs = int(X.shape[0]/6)
    X_per_band = X.reshape((n_objs,6,length)).astype(np.float32)
    X_void_per_band = X_void.reshape((n_objs,6,length)).astype(np.float32)

    vectors = np.concatenate((X_per_band,X_void_per_band),axis=1)
    print(vectors.shape)
    print(obj_ids.shape)
    print(new_targets.values.shape)
    #save relevant info int hdf5 file
    dataset = {
        "X":vectors,
        "ids":obj_ids,
        "Y": new_targets.values
    }
    save_vectors(dataset, output_dir)
    
def construct_metadata_files(m, tr_nobjs, valtest_nobjs, output_dir=defaultDir, suffix=""):
    """
    construct_metadata_files for training, validation and testing, sampling
    from the PLAsTiCC data and keeping the imbalance (sort of). 
    training is half the data_volume size, while val and test are 1/4 each (ish).
    It wirtes the new metadata files to output_dir, with a suffix if specified 
    """
    train_sample = None
    val_sample = None
    test_sample = None

    for i,c in enumerate(tr_nobjs.index.values):

        tr_quantity = int(tr_nobjs.loc[c])
        val_test_quantity = int(valtest_nobjs.loc[c])
        tr_class_m = m[m["true_target"]==c]

        if tr_quantity+val_test_quantity>tr_class_m.shape[0]:
            tr_quantity = int(np.ceil(tr_quantity/2))
            val_test_quantity = tr_class_m.shape[0] - tr_quantity
        
        tr_class_sample = tr_class_m.sample(n=tr_quantity)

        if train_sample is None:
            train_sample = tr_class_sample
        else :
            train_sample = train_sample.append(tr_class_sample)
        
        train_class_sample_ids = tr_class_sample["object_id"].values
        #ensure the train values don't get repeated in the val/test sample
        m = m[~m["object_id"].isin(train_class_sample_ids)]
        
        val_test_class_m = m[m["true_target"]==c]
        val_test_class_sample = val_test_class_m.sample(n=val_test_quantity)
        val_test_class_sample_ids = val_test_class_sample["object_id"].values
        
        size =int(np.floor(val_test_class_sample_ids.shape[0]/2))
        val_class_sample = val_test_class_sample.iloc[0:size,:]
        test_class_sample = val_test_class_sample.iloc[size:size*2,:]

        if val_sample is None:
            val_sample = val_class_sample
        else :
            val_sample = val_sample.append(val_class_sample)

        if test_sample is None:
            test_sample = test_class_sample
        else :
            test_sample = test_sample.append(test_class_sample)


        print(str(i)+": did it for class "+str(c))
        print("")

    train_sample=train_sample.sort_values(by=["object_id"])
    test_sample=test_sample.sort_values(by=["object_id"])
    val_sample=val_sample.sort_values(by=["object_id"])

    print(train_sample.groupby(["true_target"]).object_id.count())
    print(val_sample.groupby(["true_target"]).object_id.count())
    print(test_sample.groupby(["true_target"]).object_id.count())

    print(train_sample.shape[0])
    print(val_sample.shape[0])
    print(test_sample.shape[0])

    train_sample.to_csv(output_dir+"train_metadata"+suffix+".csv",sep=',',header=True, index=False)
    test_sample.to_csv(output_dir+"test_metadata"+suffix+".csv",sep=',',header=True, index=False)
    val_sample.to_csv(output_dir+"val_metadata"+suffix+".csv",sep=',',header=True, index=False)

def construct_data_files(output_dir=defaultDir, suffix=""):
    """
    construct_data_files for training, validation and testing, 
    using ids sampled when constructing metadata files (so metadata for training,
    validation and test set need to exist before hand) 
    """
    train_m = pd.read_csv(output_dir+"train_metadata"+suffix+".csv")
    print(train_m.shape)
    train_ids = train_m["object_id"].values
    print(train_ids.size)

    val_m = pd.read_csv(output_dir+"val_metadata"+suffix+".csv")
    print(val_m.shape)
    val_ids = val_m["object_id"].values
    print(val_ids.size)

    test_m = pd.read_csv(output_dir+"test_metadata"+suffix+".csv")
    print(test_m.shape)
    test_ids = test_m["object_id"].values
    print(test_ids.size)

    for i in range(1,12):
        
        print("batch {}".format(i))
        batch = pd.read_csv(output_dir+'plasticc_test_set_batch{}.csv'.format(i))

        train_batch = batch[batch.object_id.isin(train_ids)]
        val_batch = batch[batch.object_id.isin(val_ids)]
        test_batch = batch[batch.object_id.isin(test_ids)]

        if i == 1:
            train_batch.to_csv(output_dir+"train_data"+suffix+".csv",sep=',',header=True, index=False)
            val_batch.to_csv(output_dir+"val_data"+suffix+".csv",sep=',',header=True, index=False)
            test_batch.to_csv(output_dir+"test_data"+suffix+".csv",sep=',',header=True, index=False)
        else :
            train_batch.to_csv(output_dir+"train_data"+suffix+".csv",sep=',',header=False, index=False, mode="a")
            val_batch.to_csv(output_dir+"val_data"+suffix+".csv",sep=',',header=False, index=False, mode="a")
            test_batch.to_csv(output_dir+"test_data"+suffix+".csv",sep=',',header=False, index=False, mode="a")
    

####MAIN###

#get command line arguments

data_volume = 200000
suffix = '2'

# #bit of code for creating metadata files
m = pd.read_csv(defaultDir+"plasticc_test_metadata.csv")
# select extragalactic objects only
m = m[m["true_target"].isin(extragalactic)]

# wdf = m[m["ddf_bool"] ==0] #select WDF survey strategy only
objs_per_class = m.groupby("true_target").count()["object_id"].rename("objs_per_class")
print(objs_per_class)
train_volume = np.ceil(data_volume/2)
# print(train_volume)

#get how many objs per class for balanced train dataset
train_opc = np.ceil(train_volume/objs_per_class.size)
train_mask = (objs_per_class>train_opc)
train_objs_per_class = objs_per_class.copy()
train_objs_per_class[train_mask] = train_opc
train_how_many_objs_per_class = train_objs_per_class
print(train_how_many_objs_per_class)

val_test_volume = np.ceil(data_volume/2)
val_test_weights = (objs_per_class/m.shape[0]).rename("weights")
# print(val_test_weights)
val_test_how_many_objs_per_class=np.ceil(val_test_weights*data_volume)
# print(val_test_volume)
print(val_test_how_many_objs_per_class)

print("constructing metadata files")
construct_metadata_files(m,train_how_many_objs_per_class,val_test_how_many_objs_per_class, suffix=suffix)

print("contructing data files")
construct_data_files(suffix=suffix)

#this bit of code is for creating the vectors with the inteprolation fast thingy
train_csv= defaultDir+"train_data"+suffix+".csv"
train_m_csv= defaultDir+"train_metadata"+suffix+".csv"
output_dir=defaultDir+"train_data_interpolated"+suffix+".h5"
create_interpolated_vectors(train_csv, train_m_csv,output_dir, 128)

val_csv= defaultDir+"val_data"+suffix+".csv"
val_m_csv= defaultDir+"val_metadata"+suffix+".csv"
output_dir=defaultDir+"val_data_interpolated"+suffix+".h5"
create_interpolated_vectors(val_csv, val_m_csv,output_dir, 128)

test_csv= defaultDir+"test_data"+suffix+".csv"
test_m_csv= defaultDir+"test_metadata"+suffix+".csv"
output_dir=defaultDir+"test_data_interpolated"+suffix+".h5"
create_interpolated_vectors(test_csv, test_m_csv,output_dir, 128)