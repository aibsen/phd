import os, sys
from threading import local

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
from experiment import Experiment
from seeded_experiment import SeededExperiment
from transformer_classifier import TSTransformerClassifier
from positional_encodings import ConvolutionalEmbedding, TimeFiLMEncoding
import torch
from analyze_simsurvey_test_results import *


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
    "patience": 5,
    "validation_step":3
}

test_data_file_template = 'real_test_'

data_reps = ['gp_careful']

embeddings = ['default', 'conv']
pos_encodings = ['default', 'fourier']
local_decoders = ['none', 'linear']
final_pools = ['last','gap','none']

for data_rep in data_reps:
    
    i=0
    
    t_sampling = True if 'uneven' in data_rep else False
    t_sampling = True
    time_dimension = True if 'uneven' in data_rep else False

    test_data_file = data_dir+test_data_file_template+'{}_3pb_30obsd.h5'.format(data_rep)
    test_dataset = LCs(lc_length, test_data_file, packed=t_sampling)
    test_dataset.load_data_into_memory()

    # print(test_dataset.X[0])
    if type(test_dataset[0][0]) is tuple:
        input_shape = test_dataset[0][0][0].shape
    else:
        input_shape = test_dataset[0][0].shape

    print(input_shape)


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
                        test_data=test_dataset
                    )

                    experiment.run_experiment(test_data_name='test_real_30days_3pb')

                    i+=1

                    del nn
                    del experiment

                    torch.cuda.empty_cache()

                    where = exp+'/seed_1772670'
                    plot_cm(where+'/result_outputs/',
                        'test_real_30days_3pb_results.csv',
                        where+'/result_outputs/test_real_30days_3pb_cm.png')


    del test_dataset

    torch.cuda.empty_cache()

