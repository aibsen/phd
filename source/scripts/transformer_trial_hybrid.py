import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
from seeded_experiment import SeededExperiment
from transformer_hybrid import TSTransformerAutoencoderHybrid
from recurrent_models import GRU1D
from positional_encodings import TimeFiLMEncoding

from analyze_simsurvey_test_results import *

results_dir = "../../results/"
data_dir = "/home/ai/phd/data/ztf/training/"
# data_dir = "/home/ai/phd/data/ztf/testing/"
exp_name = results_dir+"transformer_hybrid_ae"


lc_length = 128
batch_size = 64
use_gpu = True
lr = 1e-03
wdc = 1e-03
seeds = [1772670]
n_seeds = 1
d_model = 128

exp_params={
    "num_epochs" : 50,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "patience": 3,
    "validation_step":3,    
    "num_output_classes": 4,
    "experiment_type": 'seq2seq',
    "pick_up" : False
}


# training_data_file=data_dir+'real_data_careful.h5'
training_data_file=data_dir+'simsurvey_data_balanced_4_mag_linear_careful.h5'
train_dataset = LCs(lc_length, training_data_file)
train_dataset.load_data_into_memory()
input_shape = train_dataset[0][0].shape
# train_dataset.lens = torch.full((len(train_dataset),),lc_length)
train_dataset.packed = True

test_dataset = LCs(lc_length, data_dir+'simsurvey_test_linear_careful.h5')
test_dataset.load_data_into_memory()
test_dataset.packed=True

real_test_dataset = LCs(lc_length, data_dir+'real_test_linear_careful_3pb_30obsd.h5')
real_test_dataset.load_data_into_memory()
real_test_dataset.packed=True


positional_encodings = ['fourier', 'default']
decoders = ['linear','gru']
for dec in decoders:
    for pos in positional_encodings:


        enc_pos = None
        dec_pos = None

        if pos == 'fourier':
            enc_pos = TimeFiLMEncoding(d_model, max_len=lc_length)

        nn=TSTransformerAutoencoderHybrid(
            nhead=4,
            nlayers=1,
            encoder_positional_encoding=TimeFiLMEncoding(d_model, max_len=lc_length),
            d_model=d_model,
            decoder= dec)

        exp_params['network_model']=nn

        name = results_dir+"_dec_{}_pos_{}".format(dec,pos)


        experiment = SeededExperiment(
            exp_name=name,
            exp_params=exp_params,
            seeds=[1772670],
            train_data=train_dataset,
            test_data=train_dataset
        )

        experiment.run_experiment()

        experiment.reset_experiment_datasets(test_data=real_test_dataset)
        experiment.run_test_phase("test_real")

        # now for classification tunning
        nn.classify=True
        # nn.freeze_autoencoder()

        exp_params={
            "num_epochs" : 100,
            "learning_rate" : lr,
            "weight_decay_coefficient" : wdc,
            "use_gpu" : use_gpu,
            "batch_size" : batch_size,
            "patience": 5,
            "validation_step":3,    
            "num_output_classes": 4,
            "experiment_type": 'classification',
            "pick_up" : True,
            "network_model": nn
        }

        experiment.reset_experiment_params(exp_params)
        experiment.reset_experiment_datasets(train_dataset,test_dataset)

        experiment.run_experiment()

        where = name+'/seed_1772670'
        plot_best_val_cm_cv(where)
        plot_cm(where+'/result_outputs/',
            'test_results.csv',
            where+'/result_outputs/test_cm.png')

        experiment.reset_experiment_datasets(test_data=real_test_dataset)
        experiment.run_test_phase("test_real")
        plot_cm(where+'/result_outputs/',
            'test_real_results.csv',
            where+'/result_outputs/test_real_cm.png')


        ### and now.. did that ruin reconstruction?
        exp_params['experiment_type'] = 'seq2seq'
        nn.classify = False
        exp_params['network_model']=nn
        experiment.reset_experiment_params(exp_params)
        experiment.run_test_phase("test_after")

        experiment.reset_experiment_datasets(test_data=real_test_dataset)
        experiment.run_test_phase("test_real_after")





