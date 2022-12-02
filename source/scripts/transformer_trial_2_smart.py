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
exp_name = results_dir+"dgen2"

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
training_data_file=data_dir+'simsurvey_data_balanced_4_mag_linear_careful.h5'
train_dataset = LCs(lc_length, training_data_file)
train_dataset.load_data_into_memory()
input_shape = train_dataset[0][0].shape
# train_dataset.lens = torch.full((len(train_dataset),),lc_length)
train_dataset.packed = True

# training_data_file=data_dir+'real_data_careful.h5'
training_data_file_0=data_dir+'real_test_linear_careful_3pb_20obsd_low_prob.h5'
train_dataset_0 = LCs(lc_length, training_data_file_0)
train_dataset_0.load_data_into_memory()
input_shape = train_dataset_0[0][0].shape
# train_dataset.lens = torch.full((len(train_dataset),),lc_length)
train_dataset_0.packed = True


test_dataset = LCs(lc_length, data_dir+'simsurvey_test_linear_careful.h5')
test_dataset.load_data_into_memory()
test_dataset.packed=True

real_test_dataset = LCs(lc_length, data_dir+'real_test_linear_careful_3pb_30obsd.h5')
real_test_dataset.load_data_into_memory()
real_test_dataset.packed=True

# epochs = [20]
# epochs = [50, 100]
# lrates = [1e-03]
# lrates = [1e-03, 1e-04, 1e-02]
# positional_encodings = ['fourier']
# freeze = ['freeze_vae', 'freeze_dec', 'no_freeze']
# freeze = ['freeze_dec','freeze_vae']
# freeze = ['no_freeze']
k1s = [0.1]
k1s_low = [0.01]


freeze_latent = ['freeze_latent']
# k1s = [0.5,0.3,0.1]
d_models = [128]
lrs = [1e-3]
e_low = [50]
fine_tune_bool = [False]


for lr in lrs:
    for heads in [4]:
        for d_model in d_models:
            for k1 in k1s:
                for e in e_low:
                    for k1_low in k1s_low:
                        for f in freeze_latent:
                            for fine_tune in fine_tune_bool:

                                exp_params_reconstruction['pick_up'] = False
                                exp_params_reconstruction['num_epochs'] = 100

                                enc_pos = TimeFiLMEncoding(d_model, max_len=lc_length)
                                dec_pos = TimeFiLMEncoding(d_model, max_len=lc_length)

                                nn=TSTransformerVAE(
                                    nhead=4,
                                    nlayers=1,
                                    encoder_positional_encoding=enc_pos,
                                    decoder_positional_encoding=dec_pos,
                                    d_model=d_model,
                                    max_len=lc_length,
                                    k0= 1-k1,
                                    k1 = k1,
                                    n_epochs=exp_params_reconstruction['num_epochs'],
                                    cycles=4,
                                    d_hid=64)

                                exp_params_reconstruction['network_model']=nn

                                name = exp_name+"_lr_{}_dhid_{}_k1{}".format(lr,64,k1)


                                experiment = SeededExperiment(
                                    exp_name=name,
                                    exp_params=exp_params_reconstruction,
                                    seeds=[1772670],
                                    train_data=train_dataset,
                                    test_data=test_dataset
                                )

                                #initial reconstruction with simulated data
                                if not os.path.isfile(name+'/seed_1772670/saved_models/final_model.pth.tar'):
                                    print('reconstruction from scratch...')
                                    experiment.run_experiment()
                                    experiment.reset_experiment_datasets(test_data=real_test_dataset)
                                    experiment.run_test_phase("test_real")

                                if freeze_latent:
                                    nn.freeze_latent_space()
                                else:
                                    nn.unfreeze_latent_space()

                                model_load_name_train='best_validation_model.pth.tar'
                                model_save_name_train='best_validation_model_e{}_k1l{}_{}.pth.tar'.format(e,k1_low,f)

                                model_load_name_final_train='final_model.pth.tar'
                                model_save_name_final_train='final_model_e{}_k1l{}_{}.pth.tar'.format(e,k1_low,f)

                                #follow up reconstruction with real data of low class probabilities

                                # if not os.path.isfile(name+'/seed_1772670/saved_models/'+model_save_name_final_train):
                                print('reconstruction from model that knows how to reconstruct simulated data')
                        
                                experiment.reset_experiment_datasets(train_data=train_dataset_0, test_data=test_dataset)

                                nn.reset_loss(e, k0=1-k1_low, k1=k1_low)
                                exp_params_reconstruction['network_model']=nn
                                exp_params_reconstruction['pick_up'] = True
                                exp_params_reconstruction['num_epochs'] = e

                                experiment.reset_experiment_params(exp_params_reconstruction)
                                
                                experiment.run_experiment(
                                    train_data_name='real_e{}_k1l_{}_{}_'.format(e,k1_low,f),
                                    test_data_name='test_e{}_k1l_{}_{}'.format(e,k1_low,f),
                                    model_load_name_train=model_load_name_train,
                                    model_save_name_train=model_save_name_train,
                                    model_load_name_final_train=model_load_name_final_train,
                                    model_save_name_final_train=model_save_name_final_train
                                    )

                                experiment.reset_experiment_datasets(test_data=real_test_dataset)
                                experiment.run_test_phase(
                                    'test_real_e{}_k1l_{}_{}'.format(e,k1_low,f),
                                    model_name=model_save_name_final_train)
                                
                                # #classification

                                # model_load_name_train=model_save_name_train
                                # model_load_name_final_train=model_save_name_final_train
                                # test_data_name = 'test_e{}_k1l_{}_{}'.format(e,k1_low,f)
                                # test_real_data_name = 'test_real_e{}_k1l_{}_{}'.format(e,k1_low,f)
                                # train_data_name='real_e{}_k1l_{}_{}_'.format(e,k1_low,f)
                                # nn.classify=True

                                # where = name+'/seed_1772670'

                                # #whether we should further train the network to reconginze classes or not
                                # print("classifying..")
                                # if fine_tune:
                                #     print('fine tuning with simulated data before classification')

                                #     model_save_name_train='best_validation_model_e{}_k1l{}_{}_fine_tuned.pth.tar'.format(e,k1_low,f)
                                #     model_save_name_final_train='final_model_e{}_k1l{}_{}_fine_tuned.pth.tar'.format(e,k1_low,f)
                                #     test_data_name = 'test_e{}_k1l_{}_{}_fine_tuned'.format(e,k1_low,f)
                                #     test_real_data_name = 'test_real_e{}_k1l_{}_{}_fine_tuned'.format(e,k1_low,f)
                                #     train_data_name='real_e{}_k1l_{}_{}_fine_tuned_'.format(e,k1_low,f)

                                #     exp_params_classification['network_model']=nn
                                #     experiment.reset_experiment_params(exp_params_classification)
                                #     experiment.reset_experiment_datasets(train_dataset,test_dataset)


                                #     if not os.path.isfile(name+'/seed_1772670/saved_models/'+model_save_name_final_train):
                                #         #in case I shut it down mid training, which ofc I do
                                #         nn.freeze_decoder()

                                #         experiment.run_experiment(
                                #             train_data_name=train_data_name,
                                #             test_data_name=test_data_name,
                                #             model_load_name_train=model_load_name_train,
                                #             model_save_name_train=model_save_name_train,
                                #             model_load_name_final_train=model_load_name_final_train,
                                #             model_save_name_final_train=model_save_name_final_train
                                #         )

                                #         plot_best_val_cm_cv(where,val_name=train_data_name[:-1])

                                #     else:
                                #         experiment.run_test_phase(
                                #             test_data_name,
                                #             model_name=model_save_name_final_train)

                                #     plot_cm(where+'/result_outputs/',
                                #         '{}_results.csv'.format(test_data_name),
                                #         where+'/result_outputs/{}_cm.png'.format(test_data_name))

                                #     model_load_name_train = model_save_name_train
                                #     model_load_name_final_train = model_save_name_final_train
                                
                                # else:
                                #     exp_params_classification['network_model']=nn
                                #     experiment.reset_experiment_params(exp_params_classification)
                                #     experiment.reset_experiment_datasets(train_dataset,test_dataset)

                                #     experiment.run_test_phase(
                                #         test_data_name,
                                #         model_name=model_load_name_final_train)

                                #     plot_cm(where+'/result_outputs/',
                                #         '{}_results.csv'.format(test_data_name),
                                #         where+'/result_outputs/{}_cm.png'.format(test_data_name))


                                # experiment.reset_experiment_datasets(test_data=real_test_dataset)

                                # experiment.run_test_phase(
                                #     test_real_data_name,
                                #     model_name=model_load_name_final_train)

                                # plot_cm(where+'/result_outputs/',
                                #     '{}_results.csv'.format(test_real_data_name),
                                #     where+'/result_outputs/{}_cm.png'.format(test_real_data_name))







