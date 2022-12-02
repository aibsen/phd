import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
from seeded_experiment import SeededExperiment
from transformer_vae import TSTransformerVAE
from positional_encodings import TimeFiLMEncoding

from analyze_simsurvey_test_results import *

results_dir = "../../results/"
data_dir = "/home/ai/phd/data/ztf/training/"
# data_dir = "/home/ai/phd/data/ztf/testing/"
exp_name = results_dir+"dgen0"

#0, num_epochs = 50
#1, num_epochs = 40

lc_length = 128
batch_size = 64
use_gpu = True
lr = 1e-03
wdc = 1e-03
seeds = [1772670]
n_seeds = 1
d_model = 128

exp_params_reconstruction={
    "num_epochs" : 100,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "patience": 5,
    "validation_step":3,    
    "num_output_classes": 4,
    "experiment_type": 'seq2seq',
    "pick_up" : False
}


exp_params_classification={
    "num_epochs" : 30,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "patience": 5,
    "validation_step":3,    
    "num_output_classes": 4,
    "experiment_type": 'classification',
    "pick_up" : True
}

# training_data_file=data_dir+'real_data_careful.h5'
# training_data_file=data_dir+'simsurvey_data_balanced_4_mag_linear_careful.h5'
# train_dataset = LCs(lc_length, training_data_file)
# train_dataset.load_data_into_memory()
# input_shape = train_dataset[0][0].shape
# # train_dataset.lens = torch.full((len(train_dataset),),lc_length)
# train_dataset.packed = True

# training_data_file=data_dir+'real_data_careful.h5'
training_data_file_0=data_dir+'real_test_linear_careful_3pb_20obsd_low_prob.h5'
train_dataset_0 = LCs(lc_length, training_data_file_0)
train_dataset_0.load_data_into_memory()
input_shape = train_dataset_0[0][0].shape
# train_dataset.lens = torch.full((len(train_dataset),),lc_length)
train_dataset_0.packed = True

# test_dataset = LCs(lc_length, data_dir+'simsurvey_test_linear_careful.h5')
# test_dataset.load_data_into_memory()
# test_dataset.packed=True

real_test_dataset = LCs(lc_length, data_dir+'real_test_linear_careful_3pb_30obsd.h5')
real_test_dataset.load_data_into_memory()
real_test_dataset.packed=True


k1s = [0.5]
k1s_low = [0.01]
e_low = [30,50]

c=0

for k1 in k1s:
    for e in e_low:
        for k1_low in k1s_low:

                exp_params_reconstruction['pick_up'] = True
                exp_params_reconstruction['num_epochs'] = 100

                enc_pos = TimeFiLMEncoding(d_model, max_len=lc_length)
                dec_pos = TimeFiLMEncoding(d_model, max_len=lc_length)

                nn=TSTransformerVAE(
                    nhead=4,
                    nlayers=1,
                    encoder_positional_encoding=enc_pos,
                    decoder_positional_encoding=dec_pos,
                    max_len=lc_length,
                    k0= 1-k1_low,
                    k1 = k1_low,
                    n_epochs=exp_params_reconstruction['num_epochs'],
                    cycles=4,
                    d_hid=64)

                exp_params_reconstruction['network_model']=nn

                name = exp_name+"_lr_{}_dhid_{}_k1{}".format(lr,64,k1)

                model_save_name_final_train='final_model_e{}_k1l{}_{}.pth.tar'.format(e,k1_low,'freeze_latent')


                experiment = SeededExperiment(
                    exp_name=name,
                    exp_params=exp_params_reconstruction,
                    seeds=[1772670],
                    test_data=real_test_dataset
                )

                experiment.run_prediction(data_name='dummy_data_{}'.format(c),model_name=model_save_name_final_train)

                experiment.reset_experiment_datasets(test_data=train_dataset_0)                

                experiment.run_prediction(data_name='dummy_data_{}'.format(c+1),model_name=model_save_name_final_train)

                c+=2
                # model_save_name_train='best_validation_model_e{}_k1l{}_{}.pth.tar'.format(e,k1_low,f)


                #follow up reconstruction with real data of low class probabilities


                                
