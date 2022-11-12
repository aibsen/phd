import os, sys
from threading import local

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
from experiment import Experiment
from seeded_experiment import SeededExperiment
from transformer_classifier import TSTransformerClassifier
from positional_encodings import ConvolutionalEmbedding, TimeFiLMEncoding

from analyze_simsurvey_test_results import *

import torch

results_dir = "../../results/"
data_dir = "/home/ai/phd/data/ztf/training/"
exp_name = "data_rep_exp_"


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
    "patience": 10,
    "validation_step":3
}

data_file_template = 'simsurvey_data_balanced_4_mag_'
test_data_file_template = 'simsurvey_test_'
# data_reps = ['linear','gp','uneven']
data_reps = ['gp_careful']
# ,'gp','uneven']

embeddings = ['default', 'conv']
pos_encodings = ['default', 'fourier']
local_decoders = ['none', 'linear']
final_pools = ['last','gap','none']

#a fuck ton of exps
for data_rep in data_reps:
    
    i = 0
    # t_sampling = True if 'uneven' in data_rep else False
    time_dimension = True if 'uneven' in data_rep else False
    t_sampling=True
    #load_datasets
    data_file=data_dir+data_file_template+'{}.h5'.format(data_rep)
    dataset = LCs(lc_length, data_file, packed=t_sampling)
    dataset.load_data_into_memory()

    if type(dataset[0][0]) is tuple:
        input_shape = dataset[0][0][0].shape
    else:
        input_shape = dataset[0][0].shape

    test_data_file = data_dir+test_data_file_template+'{}.h5'.format(data_rep)
    test_dataset = LCs(lc_length, test_data_file, packed=t_sampling)
    test_dataset.load_data_into_memory()



    for emb in embeddings:
        for pos_enc in pos_encodings:
            for ld in local_decoders:
                for pool in final_pools:

                    if pool == 'gap' and ld == 'linear':
                        continue

                    print("running experiment {} for {} data rep ...".format(i, data_rep)) 

                    embedding = None if emb == 'default' else ConvolutionalEmbedding(d_model, input_shape[0])
                    pos = None if pos_enc == 'default' else TimeFiLMEncoding(d_model, max_len=lc_length)
                    local_decoder = None if ld == 'none' else torch.nn.Linear(d_model,d_model)
                    classifier = torch.nn.Linear(d_model*lc_length,num_classes) if pool == 'none' else None

                    nn = TSTransformerClassifier(input_features=input_shape[0], 
                        max_len=lc_length, 
                        nhead=4,
                        nlayers=1,
                        uneven_t=t_sampling,
                        time_dimension = time_dimension,
                        d_model=d_model,
                        embedding_layer=embedding,
                        positional_encoding = pos,
                        local_decoder=local_decoder,
                        classifier=classifier,
                        reduction=pool
                        )


                    exp_params['network_model'] = nn

                    exp = results_dir+exp_name+data_rep+'_'+str(i)

                    experiment = SeededExperiment(
                        exp,
                        exp_params = exp_params,
                        seeds = seeds,
                        train_data=dataset,
                        test_data=test_dataset
                    )

                    experiment.run_experiment()

                    i+=1

                    del nn
                    del experiment

                    torch.cuda.empty_cache()

                    where = exp+'/seed_1772670'
                    plot_best_val_cm_cv(where)
                    plot_cm(where+'/result_outputs/',
                        'test_results.csv',
                        where+'/result_outputs/test_cm.png')


    del dataset
    del test_dataset

    torch.cuda.empty_cache()


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
    #51 again cos seems to work better
    #6linear emb, fourier, gap
    #7 conv fourier gap
    #8conv fourier last
    #9 linear, fourier, none 


