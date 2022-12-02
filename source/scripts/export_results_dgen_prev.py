import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results_dir = "../../results/"
exp_name = 'dgen0_lr_0.001_dhid_64_k10.5/result_outputs'

exp_types = ['reconstruction','classification']
e_low = [30, 50, 100]
k1s_low = [0.01, 0.1, 0.3]
freeze_latent = ['freeze_latent', 'no_freeze_latent']
fine_tune_bool = [True,False]

hyper_parameters = ['e_low','k1_low','freeze_latent']

reconstruction_ds = ['real','test_real','test']
reconstruction_metrics = hyper_parameters+['n_e']+[ d+'_loss' for d in reconstruction_ds] 
reconstruction_dict = {k:[] for k in reconstruction_metrics}

classification_ds = ['sim','test','test_real']
classification_metrics_one = ['f1','accuracy','loss']
classification_metrics =hyper_parameters+['n_e']+[d+"_"+m for d in classification_ds[1:] for m in classification_metrics_one]\
    +[d+"_"+m+"_fine_tuned" for d in classification_ds for m in classification_metrics_one]
classification_dict = {k: [] for k in classification_metrics}

for exp_type in exp_types:
    for e in e_low:
        for k1l in k1s_low:
            for f in freeze_latent:

                row = {k:None for k in reconstruction_metrics} if exp_type == 'reconstruction'\
                    else {k:None for k in classification_metrics}

                row['e_low'] = e
                row['k1_low'] = k1l
                row['freeze_latent'] = f

                if exp_type == 'reconstruction':

                    for ds in reconstruction_ds: 
                        
                        f_ds = '_validation' if ds == 'real' else ""
                        f_name=results_dir+exp_name+"/"\
                            +"reconstruction_{}_e{}_k1l_{}_{}{}_summary.csv".format(ds,e,k1l,f,f_ds)

                        print(f_name)
                        df = pd.read_csv(f_name)

                        if ds == 'real':
                            row['n_e'] = df.mean_epoch.values[0]
                        row['{}_loss'.format(ds)] = df.mean_loss.values[0]

                    for k,v in row.items():
                        reconstruction_dict[k].append(v)
                    
                elif exp_type == 'classification':


                    for ds in classification_ds:

                        f_ds = '_validation' if ds=='sim' else ""
                        d = 'real' if ds=='sim' else ds

                        if ds == 'sim':
                            f_name=results_dir+exp_name+"/"\
                                +"{}_e{}_k1l_{}_{}_fine_tuned{}_summary.csv".format(d,e,k1l,f,f_ds)
                            print(f_name)
                            
                            df = pd.read_csv(f_name) 
                            row['n_e']= df.mean_epoch.values[0]
                            
                            for k in classification_metrics_one:
                                row[ds+'_'+k+'_fine_tuned'] = df['mean_{}'.format(k)].values[0]
                        
                        else:
                            for fine_tune in fine_tune_bool:
                                fine_tune_name = 'fine_tuned_'if fine_tune else ''
                                f_name=results_dir+exp_name+"/"\
                                    +"{}_e{}_k1l_{}_{}{}_{}summary.csv".format(d,e,k1l,f,f_ds,fine_tune_name)
                                print(f_name)

                                df = pd.read_csv(f_name)
                                fs = "_fine_tuned" if fine_tune else ''

                                for k in classification_metrics_one:
                                    row[ds+'_'+k+fs] = df['mean_{}'.format(k)].values[0]

                    for k,v in row.items():
                        classification_dict[k].append(v)

rec_df = pd.DataFrame.from_dict(reconstruction_dict)
class_df = pd.DataFrame.from_dict(classification_dict)
all_results = class_df.merge(rec_df, on=hyper_parameters,suffixes=('', '_reconstruction'))

all_results.to_csv('dgen0_prev_results.csv',index=False)

                    