import os, sys
from threading import local

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
from experiment import Experiment
from seeded_experiment import SeededExperiment
from transformer_classifier import TSTransformerClassifier
from positional_encodings import ConvolutionalEmbedding, TimeFiLMEncoding
import torch

results_dir = "../../results/"
data_dir = "/home/ai/phd/data/ztf/training/"
exp_name = "data_rep_exp_5_"


lc_length = 128
d_model = 120
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
data_reps = ['linear','gp','uneven','uneven_tnorm']
# data_reps = ['uneven','uneven_tnorm']
data_reps = ['uneven_tnorm_back']

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


    # t_sampling = False

    # classifier = torch.nn.Linear(d_model*lc_length,num_classes)
    # embedding = ConvolutionalEmbedding(d_model, input_shape[0])
    pos = TimeFiLMEncoding(d_model, max_len=lc_length)


    nn = TSTransformerClassifier(input_features=input_shape[0], 
        max_len=lc_length, 
        nhead=4,
        nlayers=1,
        uneven_t=t_sampling,
        time_dimension = time_dimension,
        positional_encoding = pos,
        d_model=d_model,
        # reduction='gap'
        # embedding_layer=embedding
        # classifier=classifier
        )


    exp_params['network_model'] = nn
    # experiment = Experiment(
    #     nn,
    #     experiment_name=exp_name,
    #     num_epochs=num_epochs,
    #     train_data=dataset,
    #     val_data=dataset #trivial case to check if things work
    # )

    experiment = SeededExperiment(
        results_dir+exp_name+data_rep,
        exp_params = exp_params,
        seeds = seeds,
        train_data=dataset
    )

    experiment.run_experiment()
    #2 last layer uses all zs
    #20 same but with local decoder 
    #3 last layer is gap
    #30 last layer is gap, but maybe there was a mistake
    #31 for uneven considers correct lengths for pooling
    #32 for uneven considers correct lengths for pooling, maybe there was a mistake
    #33 for uneven regular pooling, x completitud
    #34 added t_dim for uneven regular pooling, x completitud 2, why did it drop from 3-30?
    #4 gap, conv embedding
    #5linear embedding, fourier pos, last
    #50 mask indices from l instead of l+1
    #linear emb, fourier, gap



