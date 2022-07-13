import pandas as pd
import h5py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
import numpy as np
import torch
from seeded_experiment import SeededExperiment
from plot_utils import *

train_metadata_dir = "../../data/plasticc/csvs/"
train_data_dir = "../../data/plasticc/dmdt/training/"
results_dir = "../../results/"
seeds = [1772670]

def why_are_ids_changing():

    #because I was saving them as small ints
    for resolution in [24,32,40]:
        for noise_code in [2,3,4,5,6,7]:
            print("")
            print("Resolution {}, noise code {}".format(resolution,noise_code))
            train_metadata_file = "plasticc_train_metadata_augmented_noise{}.csv".format(noise_code)
            train_data_file = "plasticc_train_lightcurves_augmented_noise{}.csv".format(noise_code)
            train_dataset_file = "dmdts_training_{}x{}_b00augmented_noise{}.h5".format(resolution,resolution,noise_code) 

            train_metadata = pd.read_csv(train_metadata_dir+train_metadata_file)
            train_data = pd.read_csv(train_metadata_dir+train_data_file)

            negative_id_count = train_metadata[train_metadata.object_id < 0].count().object_id
            negative_id_count0 = train_data[train_data.object_id < 0].count().object_id
            print(negative_id_count)
            print(negative_id_count0)

            with h5py.File(train_data_dir+train_dataset_file,'r') as f:
                ids = f["ids"]
                np_ids = np.array(ids)
                np_negative_ids = np.where(np_ids<0)
                print(len(np_negative_ids[0]))
                print(np_negative_ids)
                # print(np_ids[0])

                # torch_ids = torch.tensor(ids,device=torch.device('cuda'), dtype = torch.int)
                # torch_negative_ids = torch.where(torch_ids<0)
                # print(len(torch_negative_ids[0]))

                torch_ids_cpu = torch.tensor(ids,device=torch.device('cpu'), dtype = torch.int)
                torch_negative_ids_cpu = torch.where(torch_ids_cpu<0)
                print(len(torch_negative_ids_cpu[0]))
                one_negative_id_index = torch_negative_ids_cpu[0][0]


                print("original int")
                print(int(np_ids[one_negative_id_index]))
                print("when converted to torch int")
                print(torch_ids_cpu[one_negative_id_index])
                print("when converted to torch long")
                print(torch_ids_cpu[one_negative_id_index].long())

            # train_dataset = LCs(24,train_data_dir+train_dataset_file,n_channels=6)
            # train_dataset.load_data_into_memory()
            # ids = train_dataset.ids.cpu().numpy()
            # print(ids)
            # negative_ids = np.where(ids<0)[0]
            # print(negative_ids)
            # print(len(negative_ids))

                f.close()
            break
        break

def save_best_f1(exp_name):

    experiment = SeededExperiment(
        exp_name = exp_name,
        seeds = seeds
    )
    experiment.save_best_fold_results()

def object_count(data_file):
    data = pd.read_csv(data_file)
    print(len(data.object_id.unique()))
# for r in [24]:
#     # for n in [2,3,4,5,6,7]:
# # for r in [24]:
#     for arch in ['0']:
#         for code in ['00']:
#     # for code in ['00']:
#             e = '{}_{}x{}final'.format(arch,r,r,code)
#         # e = '1_{}x{}_b00augmented_noise{}'.format(r,r,n)
#             exp_name = results_dir+e
#             save_best_f1(exp_name)

# final_dir = "../../results/2_24x24final/result_outputs/test_4_results.csv"
# results = pd.read_csv(final_dir)
# prediction = results.prediction
# target = results.target
# plot_best_val_cm(target,prediction,save=True, output_file="../../results/2_24x24final/result_outputs/test_4_cm.png")

object_count(train_metadata_dir+'plasticc_new10kshuffle_train_metadata.csv')