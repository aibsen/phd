import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
from experiment import Experiment
from seeded_experiment import SeededExperiment
from transformer_classifier import TSTransformerClassifier
import torch

results_dir = "../../results/"
data_dir = "/home/ai/phd/data/ztf/training/"
exp_name = "data_rep_exp_2_"


lc_length = 128
d_model = 128
batch_size = 64
num_epochs = 100
use_gpu = True
lr = 1e-03
wdc = 1e-03
seeds = [1772670]
n_seeds = 1
num_classes = 4

exp_params={
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "num_output_classes": num_classes,
    "patience": 5,
    "validation_step":3
}

data_file_template = 'simsurvey_data_balanced_4_mag_'
test_data_file_template = 'simsurvey_test_'
# data_reps = ['linear','gp','uneven','uneven_tnorm']
data_reps = ['uneven_tnorm_backl']
# ,'uneven_tnorm']
# data_reps = ['uneven','uneven_tnorm']

for data_rep in data_reps:

    t_sampling = True if 'uneven' in data_rep else False
    time_dimension = True if 'uneven' in data_rep else False

    data_file=data_dir+data_file_template+'{}.h5'.format(data_rep)
    dataset = LCs(lc_length, data_file, packed=t_sampling)
    dataset.load_data_into_memory()
    # print(dataset.lens)
    # print(dataset[0])
    # print(dataset[0][0] is tuple)
    if type(dataset[0][0]) is tuple:
        input_shape = dataset[0][0][0].shape
    else:
        input_shape = dataset[0][0].shape

    print(input_shape)
    # .shape if t_sampling else dataset[0][0].shape

    # loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_data_file = data_dir+test_data_file_template+'{}.h5'.format(data_rep)
    test_dataset = LCs(lc_length, test_data_file, packed=t_sampling)
    test_dataset.load_data_into_memory()


    # t_sampling = True
    # time_dimension = True

    nn = TSTransformerClassifier(input_features=input_shape[0], 
        max_len=lc_length, 
        nhead=4,
        nlayers=1,
        uneven_t=t_sampling,
        time_dimension=time_dimension,
        d_model=d_model)

    exp_params['network_model'] = nn
    # experiment = Experiment(
    #     nn,
    #     experiment_name=exp_name,
    #     num_epochs=num_epochs,
    #     train_data=dataset,
    #     val_data=dataset #trivial case to check if things work
    # )

    experiment = SeededExperiment(
        results_dir+exp_name+data_rep+'0',#2 = no local_encoder, lr 03, #3 lr 04
        exp_params = exp_params,#4 lr04 wdc02
        seeds = seeds,#5 unvenen pos
        train_data=dataset,#6 disregar tdim and uneven pos
        test_data = test_dataset
    )#7,6 but with gather
    #8,7 but correct padding
    #9, 8 but without t dim
    #10 all correct, no local decoder
    #11 default pos not considerint t_dim, correct padding and gather

    experiment.run_experiment()






