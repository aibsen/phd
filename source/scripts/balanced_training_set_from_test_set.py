import pandas as pd
import numpy as np

plasticc_data_dir = "../../data/plasticc/csvs/" 
test_metadata_file =  plasticc_data_dir+'plasticc_test_metadata.csv'

extragalactic = [90,67,52,42,62,95]
# ,15,64,88]
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

def separate_files():
    metadata = pd.read_csv(test_metadata_file)
    metadata = metadata[metadata.true_target.isin(extragalactic)]
    train_volume = 200000
    targets = metadata.true_target.unique()
    training_ids = []
    print(targets)
    required = int(train_volume/len(targets))

    for target in targets:
        typed_metadata = metadata[metadata.true_target == target]
        n_samples = typed_metadata.shape[0]
        sample = typed_metadata.sample(int(n_samples/2)) if n_samples < required else typed_metadata.sample(required)
        training_ids += sample.object_id.tolist()

    condition = metadata.object_id.isin(training_ids)
    training_metadata = metadata[condition]
    training_metadata.to_csv(plasticc_data_dir+'plasticc_balanced_train_metadata_eg.csv')

    test_metadata = metadata[~condition]
    test_metadata.to_csv(plasticc_data_dir+'plasticc_balanced_test_metadata_eg.csv')

    training_data = None

    for i in range(1,12):
        data = pd.read_csv(plasticc_data_dir+'plasticc_test_set_batch{}.csv'.format(i))
        print(data.shape)
        condition = data.object_id.isin(training_ids)
        new_training_data = data[condition]
        training_data = new_training_data if training_data is None else pd.concat([training_data,new_training_data], ignore_index=True)
        test_data = data[~condition]
        test_data.to_csv(plasticc_data_dir+'plasticc_balanced_test_set_batch{}_eg.csv'.format(i))
        print(training_data.shape)
        print(test_data.shape)
        print("")

    training_data.to_csv(plasticc_data_dir+"plasticc_balanced_train_data_eg.csv")

    
separate_files()

def shuffle(data_file, metadata_file):
    metadata = pd.read_csv(metadata_file)
    print(metadata.head(20).true_target)
    data = pd.read_csv(data_file)
    print(data.head(20))
    metadata = metadata.sample(frac=1).reset_index(drop=True)
    print(metadata.head(20).true_target)
    data = data[data.object_id.isin(metadata.object_id)]
    print(data.head(20))

shuffle(plasticc_data_dir+"plasticc_balanced_train_data_eg.csv",plasticc_data_dir+'plasticc_balanced_train_metadata_eg.csv')