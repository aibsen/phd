from tkinter import E
from urllib.request import urlopen
from html.parser import HTMLParser
import csv
import os
import pandas as pd
from astropy.time import Time
import numpy as np

data_dir = "../../data/ztf/tns/csvs/2022-08/"

metadata_fn_tns = ["agn","slsn-1","slsn-2","sn1a-91bg","sn1a-91T",'sn1a-CSM',
    'sn1a-pec','sn1a-SC','sn1a','sn1ax','sn1b-c','sn2','sn2l','tde']
# metadata_fn_tns = 'agn'
metadata_fn_alerce = "alerce_sn"

output_path='alerce_sn_lc.csv'

url_template_mars = r"https://mars.lco.global/?sort_value=jd&sort_order=desc&objectId="
url_template_lasair = r"https://lasair-ztf.lsst.ac.uk/object/"
keys_mars = ["objectId","time","filter","magpsf"]
keys_lasair = ["MJD",'Filter', 'magpsf','status']

def scrap_sns_mars(metadata_fn, url_template=url_template_mars, keys=keys_mars, output_path=output_path):
    print(metadata_fn)
    metadata = pd.read_csv(data_dir+metadata_fn+'.csv')
    ids = list(metadata.objectId.values)
    n_ids = len(ids)
    for i, id in enumerate(ids):
        print('Object {}/{}: {}'.format(i,n_ids,id))
        try:
            url = url_template+id
            tables = pd.read_html(url) # Returns list of all tables on page
            print(tables)
            df = tables[0][keys]
            print(df)
            times = list(df.time.astype('str').values)
            times = Time(times, format="iso")
            mjds = times.mjd
            df.loc[:,'time']=mjds
            df.to_csv(data_dir+output_path, mode='a', header=not os.path.exists(output_path),index=False)
        except Exception as e:
            print('Object id {} not found in url'.format(id))
            print(e)

def scrap_sns_lasair(metadata_fn, url_template=url_template_lasair, keys=keys_lasair, output_path=output_path):
    print(metadata_fn)
    metadata = pd.read_csv(data_dir+metadata_fn+'.csv')
    ids = list(metadata.objectId.values)
    n_ids = len(ids)
    for i, id in enumerate(ids):
        print('Object {}/{}: {}'.format(i,n_ids,id))
        try:
            url = url_template+id
            tables = pd.read_html(url) # Returns list of all tables on page
            df = tables[1][keys]
            df.loc[:,"objectId"]=id
            df.loc[:,'split'] = df.loc[:,'magpsf'].apply(lambda x: x.split(" "))
            df.loc[:,'magpsf'] = df.loc[:,'split'].apply(lambda x: x[0])
            df.loc[:,'magpsf_error'] = df.loc[:,'split'].apply(lambda x: x[-1] if len(x)>1 else np.nan)
            df = df[["objectId"]+keys_lasair+['magpsf_error']]
            # (r=0, g=1)
            df.loc[:,'Filter'] = df.loc[:,'Filter'].apply(lambda x: 0 if x =='r' else 1)
            df.loc[:,'status'] = df.loc[:,'status'].apply(lambda x: 0 if x == 'non-detection' else 1)
            df = df.rename(columns={'MJD':"time","status":"detected","Filter":"filter"})
            df.to_csv(data_dir+output_path, mode='a', header=not os.path.exists(output_path),index=False)
        except Exception as e:
            print('Object id {} not found in url'.format(id))
            print(e)

def how_many_objects(metadata):
    print("Total objects: "+str(metadata.shape[0]) )
    classes = str(metadata['predicted_class'].unique())
    print("{} Classes: {}".format(len(classes),classes))
    objs_per_class = metadata.groupby("predicted_class").count()
    print(objs_per_class)


def clean_tns_metadata(metadata_fn):
    metadata = pd.read_csv(data_dir+metadata_fn+'-dirty.csv')
    print(metadata.shape)
    # print(metadata.keys())
    metadata = metadata[['Name','RA','DEC','Obj. Type', 'Redshift', 'Host Name', 'Host Redshift','Reporting Group/s','Classifying Group/s', 'Disc. Internal Name','Sender']]
    metadata = metadata.rename(columns={'Obj. Type': 'predicted_class', 'Disc. Internal Name':'objectId','Classifying Group/s':'classified_by',
        'Reporting Group/s':'reported_by','Sender':'sent_by'})
    # # # metadata = metadata.rename(columns={'objectId':'predicted_class', 'predicted_class':'objectId'})
    # print(metadata.head())
    metadata.to_csv(data_dir+metadata_fn+'.csv', index=False)

def not_crossing(metadata_tns_fn, metadata_alerce_fn):
    metadata_tns = pd.read_csv(data_dir+metadata_tns_fn+'.csv')
    metadata_alerce = pd.read_csv(data_dir+metadata_alerce_fn)
    print(metadata_tns.shape)
    crossing = metadata_tns.objectId.isin(metadata_alerce.objectId)
    # print(crossing.shape)
    not_crossing = metadata_tns[~crossing]
    print(not_crossing.shape)
    not_crossing.to_csv(data_dir+metadata_tns_fn+'.csv', index=False)

def merge_files(metadata_fn,n_files):
    metadata = []
    for i in range(n_files):
        m= pd.read_csv(data_dir+metadata_fn+'-{}.csv'.format(i))
        metadata.append(m)
        print(i)
    all_metadata = pd.concat(metadata,axis=0,ignore_index=True)
    all_metadata.to_csv(data_dir+metadata_fn+'.csv',sep=',',index=False)



# merge_files(metadata_fn_tns,2)
# for metadata_fn in metadata_fn_tns:
# print(metadata_fn_alerce)
# clean_tns_metadata(metadata_fn_tns)
    # not_crossing(metadata_fn,metadata_fn_alerce)
scrap_sns_lasair(metadata_fn_alerce,output_path='{}_lc.csv'.format(metadata_fn_alerce))
    # scrap_sns_lasair(metadata_fn,output_path='{}_lc.csv'.format(metadata_fn))
    # metadata_tns = pd.read_csv(data_dir+metadata_fn_tns+'.csv')
    # 
# how_many_objects(metadata_tns)
