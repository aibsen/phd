from cProfile import run
import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess_data_utils
from utils import simsurvey_ztf_type_dict

runs = 10
mag_cut = 19
csv_data_dir = '../../data/ztf/csv/simsurvey/'
interpolated_data_dir = '../../data/ztf/training/linearly_interpolated/'

def merge_files():
    datas = []
    metadatas = []

    for run_code in range(runs):
        print(run_code)
        print("")
        for k, type_name in simsurvey_ztf_type_dict.items():
            
            data_fn = csv_data_dir+"lcs_{}_{}_{}.csv".format(type_name,run_code,mag_cut)
            data = pd.read_csv(data_fn)

            metadata_fn = csv_data_dir+"meta_{}_{}_{}.csv".format(type_name,run_code,mag_cut)
            metadata = pd.read_csv(metadata_fn)

            print(type_name)
            print(metadata.shape)
            print("")
    
            # print(metadata)

            true_target = int(k) if int(k)<6 else 3 #IIP and IIn are clumped into one class
            metadata['true_target'] = np.full(metadata.shape[0],true_target)
            datas.append(data)
            metadatas.append(metadata)
        print("")

    simsurvey_lcs = pd.concat(datas, axis=0, ignore_index=True)
    simsurvey_meta = pd.concat(metadatas, axis=0, ignore_index=True)
    print(simsurvey_meta.shape)
    print(simsurvey_meta.object_id.unique().shape)
    print(simsurvey_lcs.shape)
    print(simsurvey_lcs.object_id.unique().shape)

    # simsurvey_lcs.to_csv(csv_data_dir+"simsurvey_lcs_20.csv",index=False)
    # simsurvey_meta.to_csv(csv_data_dir+"simsurvey_metadata_20.csv",index=False)


def sanity_check():
    data_fn = csv_data_dir+"simsurvey_lcs_20.csv"
    metadata_fn = csv_data_dir+"simsurvey_metadata_20.csv"

    data = pd.read_csv(data_fn)
    metadata = pd.read_csv(metadata_fn)

    print((data.object_id.unique() == metadata.object_id.unique()).all())
    print(data.object_id.unique().shape == metadata.object_id.unique().shape)
    print(data.object_id.unique().shape)

def create_interpolated_vectors(data_fn, meta_fn, output_fn):

    data_fn = csv_data_dir+data_fn
    meta_fn = csv_data_dir+meta_fn

    data = pd.read_csv(data_fn)
    metadata = pd.read_csv(meta_fn)

    X,ids = preprocess_data_utils.create_interpolated_vectors(data,128,n_channels=2)
    tags = metadata[metadata.object_id.isin(ids)].sort_values(['object_id'])
    print(tags)
    Y = tags.true_target.values

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

    preprocess_data_utils.save_vectors(dataset,interpolated_data_dir+output_fn)


n_runs_dict = {
    'Ia': 4,
    'Ia-91bg': 70,
    'Iax': 108,
    'IIP': 2,
    'Ibc':3,
    'SLSN':23,
    'IIn':2

}

def create_balanced_dataset_zcare(max_runs=108):
    datas = []
    metadatas = []

    test_datas = []
    test_metadatas = []

    def f(k,v):
        if k == 'IIP': return int(np.ceil(2000/v))
        elif k == 'IIn': return int(np.ceil(800/v))
        else: return int(np.ceil(2800/v))

    samples_per_run = {k: f(k,v) for k,v in n_runs_dict.items()}
    
    print(samples_per_run)
    stop = {k:0 for k in n_runs_dict}
    missing_samples = {k:0 for k in n_runs_dict}
    

    for run_code in range(max_runs):
        for k, type_name in simsurvey_ztf_type_dict.items():
            if stop[type_name] <=2100 and run_code<n_runs_dict[type_name]:
                # print(run_code)
                # print(type_name)

                true_target = int(k) if int(k)<6 else 3 #IIP and IIn are clumped into one class

                print('run {}, type {}'.format(str(run_code),type_name))

                data_fn = csv_data_dir+"zcare_lcs_{}_{}.csv".format(type_name,run_code)
                data = pd.read_csv(data_fn)

                metadata_fn = csv_data_dir+"zcare_meta_{}_{}.csv".format(type_name,run_code)
                metadata = pd.read_csv(metadata_fn)

                n = samples_per_run[type_name]+missing_samples[type_name]

                try:
                    m_sample = metadata.sample(n=n,ignore_index=True).sort_values('object_id')
                    missing_samples[type_name]=0
                except ValueError as ve:
                    print("probs tried to sample more than we had")
                    print(ve)
                    print("taking everything available instead")
                    m_sample = metadata
                    missing_samples[type_name]=n-m_sample.shape[0]

                
                m_sample['true_target'] = np.full(m_sample.shape[0],true_target)

                m_train = m_sample.iloc[:3*int(n/4):,:]
                m_test = m_sample.iloc[3*int(n/4):,:]

                d_train = data[data.object_id.isin(m_train.object_id)]
                d_test = data[data.object_id.isin(m_test.object_id)]
                #no extra runs for these classes, need to sample

                test_datas.append(d_test)
                test_metadatas.append(m_test)    
                datas.append(d_train)
                metadatas.append(m_train)

                stop[type_name] += m_train.shape[0]

                #sanity check
                try:
                    assert((m_train.object_id.unique()==d_train.object_id.unique()).all())
                    assert((m_test.object_id.unique()==d_test.object_id.unique()).all())
                except AssertionError as ae:
                    print("repeated ids probably")
                    sys.exit(1)
            
    simsurvey_lcs = pd.concat(datas, axis=0, ignore_index=True)
    simsurvey_meta = pd.concat(metadatas, axis=0, ignore_index=True)

    simsurvey_lcs_test = pd.concat(test_datas, axis=0, ignore_index=True)
    simsurvey_meta_test = pd.concat(test_metadatas, axis=0, ignore_index=True)

    try:
        #check that ids match
        assert((simsurvey_lcs.object_id.unique()==simsurvey_meta.object_id.unique()).all())
        assert((simsurvey_lcs_test.object_id.unique()==simsurvey_meta_test.object_id.unique()).all())
        #check that ids dont repeat
        assert(simsurvey_meta.object_id.shape[0]==simsurvey_meta.object_id.unique().shape[0])
        assert(simsurvey_meta_test.object_id.shape[0]==simsurvey_meta_test.object_id.unique().shape[0])

        print(simsurvey_meta.groupby('true_target').count()['object_id'])
        print(simsurvey_meta_test.groupby('true_target').count()['object_id'])
        simsurvey_lcs.to_csv(csv_data_dir+"simsurvey_lcs_balanced_zcare.csv",index=False)
        simsurvey_lcs_test.to_csv(csv_data_dir+"simsurvey_lcs_balanced_zcare_test.csv",index=False)
        simsurvey_meta.to_csv(csv_data_dir+"simsurvey_metadata_balanced_zcare.csv",index=False)
        simsurvey_meta_test.to_csv(csv_data_dir+"simsurvey_metadata_balanced_zcare_test.csv",index=False)

    except AssertionError as ae:
        print("sanity check failed")
        print(ae)
        print("n_objects: {}".format(simsurvey_meta.object_id.shape[0]))
        print("unique ids: {}".format(simsurvey_meta.object_id.unique().shape[0]))
        simsurvey_meta['duplicates'] = simsurvey_meta.duplicated(subset='object_id',keep=False)
        duplicates = simsurvey_meta[simsurvey_meta['duplicates']==True]        
        print("n duplicates: {}".format(duplicates.shape[0]))
        print("n duplicated ids: {}".format(duplicates.object_id.unique().shape[0]))
        print("classes that have duplicates: {}".format(duplicates.true_target.unique()))
        print(duplicates)


def create_balanced_dataset():
    datas = []
    metadatas = []

    test_datas = []
    test_metadatas = []

    for run_code in range(runs):
        for k, type_name in simsurvey_ztf_type_dict.items():
            print(run_code)
            print(type_name)

            true_target = int(k) if int(k)<6 else 3 #IIP and IIn are clumped into one class

            #no extra runs for these classes, need to sample
            if ((type_name == 'Ia' 
                or type_name == 'Ibc' 
                or type_name == 'IIP'
                or type_name == 'IIn')  and run_code<10): 
                
                print('run {}, type {}'.format(str(run_code),type_name))

                # mag_cut = 19 if type_name == 'Ia' else 20
                #sample 280 per class, except for SNII, divide between IIP and IIn
                n = 280 if true_target != 3 else (205 if type_name == 'IIP' else 75) 
                
                data_fn = csv_data_dir+"zcare_lcs_{}_{}.csv".format(type_name,run_code)
                data = pd.read_csv(data_fn)

                metadata_fn = csv_data_dir+"zcare_meta_{}_{}.csv".format(type_name,run_code,mag_cut)
                metadata = pd.read_csv(metadata_fn)

                try:
                    m_sample = metadata.sample(n=n,ignore_index=True).sort_values('object_id')
                except ValueError as ve:
                    print("probs tried to sample more than we had")
                    print(ve)
                    print("taking everything available instead")
                    m_sample = metadata

                d_sample = data[data.object_id.isin(m_sample.object_id)]

                m_sample['true_target'] = np.full(m_sample.shape[0],true_target)

                if run_code>7:
                    test_datas.append(d_sample)
                    test_metadatas.append(m_sample)    

                else:
                    datas.append(d_sample)
                    metadatas.append(m_sample)

           #for this classes we need all the sims we have
            elif (((type_name == 'Iax') and run_code < 56) 
                or ((type_name == 'Ia-91bg') and run_code < 52)
                or ((type_name == 'SLSN') and run_code < 170)):

                print('run {}, type {}'.format(str(run_code),type_name))
                #extra run files have different name format and numbering
                data_fn_template = 'lcs_' if run_code<10 else 'extra_lcs_'
                meta_fn_template =  'meta_' if run_code<10 else 'extra_meta_'
                r = run_code if run_code < 10 else run_code-10

                mag_cut = 20

                data_fn = csv_data_dir+data_fn_template+"{}_{}_{}.csv".format(type_name,r,mag_cut)
                d_sample = pd.read_csv(data_fn)

                metadata_fn = csv_data_dir+meta_fn_template+"{}_{}_{}.csv".format(type_name,r,mag_cut)
                m_sample = pd.read_csv(metadata_fn)
                m_sample['true_target'] = np.full(m_sample.shape[0],true_target)


                if (((type_name == 'Iax') and run_code >=46)
                    or ((type_name == 'Ia-91bg') and run_code >=42)
                    or ((type_name == 'SLSN') and run_code >=130)):
                    test_datas.append(d_sample)
                    test_metadatas.append(m_sample)
                else:
                    print("ELSE")
                    metadatas.append(m_sample)
                    datas.append(d_sample) 

            else:
                continue

            #sanity check
            try:
                assert((m_sample.object_id.unique()==d_sample.object_id.unique()).all())
            # print(m_sample.object_id.unique()==d_sample.object_id.unique())
            except AssertionError as ae:
                print("repeated ids probably")
                print("n objects: {}".format(m_sample.object_id.shape))
                print("unique ids: {}".format(d_sample.object_id.unique().shape))
                sys.exit(1)
            
    simsurvey_lcs = pd.concat(datas, axis=0, ignore_index=True)
    simsurvey_meta = pd.concat(metadatas, axis=0, ignore_index=True)

    simsurvey_lcs_test = pd.concat(test_datas, axis=0, ignore_index=True)
    simsurvey_meta_test = pd.concat(test_metadatas, axis=0, ignore_index=True)

    try:
        #check that ids match
        assert((simsurvey_lcs.object_id.unique()==simsurvey_meta.object_id.unique()).all())
        assert((simsurvey_lcs_test.object_id.unique()==simsurvey_meta_test.object_id.unique()).all())
        #check that ids dont repeat
        assert(simsurvey_meta.object_id.shape[0]==simsurvey_meta.object_id.unique().shape[0])
        assert(simsurvey_meta_test.object_id.shape[0]==simsurvey_meta_test.object_id.unique().shape[0])

        print(simsurvey_meta.groupby('true_target').count()['object_id'])
        print(simsurvey_meta_test.groupby('true_target').count()['object_id'])
        simsurvey_lcs.to_csv(csv_data_dir+"simsurvey_lcs_balanced_1.csv",index=False)
        simsurvey_lcs_test.to_csv(csv_data_dir+"simsurvey_lcs_balanced_1_test.csv",index=False)
        simsurvey_meta.to_csv(csv_data_dir+"simsurvey_metadata_balanced_1.csv",index=False)
        simsurvey_meta_test.to_csv(csv_data_dir+"simsurvey_metadata_balanced_1_test.csv",index=False)

    except AssertionError as ae:
        print("sanity check failed")
        print(ae)
        print("n_objects: {}".format(simsurvey_meta.object_id.shape[0]))
        print("unique ids: {}".format(simsurvey_meta.object_id.unique().shape[0]))
        simsurvey_meta['duplicates'] = simsurvey_meta.duplicated(subset='object_id',keep=False)
        duplicates = simsurvey_meta[simsurvey_meta['duplicates']==True]        
        print("n duplicates: {}".format(duplicates.shape[0]))
        print("n duplicated ids: {}".format(duplicates.object_id.unique().shape[0]))
        print("classes that have duplicates: {}".format(duplicates.true_target.unique()))
        print(duplicates)

# create_balanced_dataset()
# create_balanced_dataset_zcare()
# merge_files()
# sanity_check()
create_interpolated_vectors('simsurvey_lcs_balanced_zcare.csv','simsurvey_metadata_balanced_zcare.csv','simsurvey_data_balanced_zcare.h5')
