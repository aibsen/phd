import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results_dir = "../../results/"


keys = [
    'exp_name',
    'k1', 'freeze', 'dm', 'e_r', 'lr', 'h',
    'epoch',
    'val_accuracy','val_f1','val_loss',
    'test_accuracy','test_f1','test_loss',
    'test_real_accuracy', 'test_real_f1','test_real_loss',
    'epoch_r','val_loss_r','test_loss_r','test_real_loss_r',
    'test_loss_r_after','test_real_loss_r_after'        
]

r_dict = {k:[] for k in keys}

metrics = ['accuracy', 'f1','loss']

for exp_name in os.listdir(results_dir):
    f = os.path.join(results_dir, exp_name)

    try:

        if "cyc" in f:

            summary_dir = os.path.join(f,'result_outputs')
            for summary in os.listdir(summary_dir):
                
                if 'final_training' not in summary:
                        
                        summary_path = os.path.join(summary_dir,summary)
                        df = pd.read_csv(summary_path)
                        no_extension = summary.split('_')[:-1]
                        
                        if 'reconstruction' not in no_extension[0]:
                            item = no_extension[0:]
                            ds_name = '_'.join(item) if 'validation' not in item else 'val'
                            # print(ds_name)
                            for m in metrics:
                                try:
                                    v = df['mean_'+m].values[0]
                                    r_dict[ds_name+'_'+m].append(v)

                                except KeyError as e:
                                    r_dict[ds_name+'_'+m].append(None)

                            if 'validation' in item:
                                r_dict['epoch'].append(df['mean_epoch'].values[0])

                        elif 'reconstruction' in no_extension[0]:
                            # print(no_extension)
                            item = no_extension[1:]
                            # print(item)
                            item = item if 'after' not in summary else item[:-1]
                            ds_name = '_'.join(item) if 'validation' not in item else 'val'
                            # print(ds_name)
                            try:
                                if 'after' not in summary:
                                    v = df['mean_loss'].values[0]
                                    r_dict[ds_name+'_loss_r'].append(v)
                                else: 
                                    v = df['mean_loss'].values[0]
                                    r_dict[ds_name+'_loss_r_after'].append(v)
                                
                                if 'validation' in item:
                                    r_dict['epoch_r'].append(df['mean_epoch'].values[0])

                                    
                            
                            except KeyError as e:
                                ke = ds_name+'_loss_r' if 'after' not in summary else ds_name+'_loss_r_after'
                                # print(ke)
                                # print(r_dict.keys())
                                r_dict[ke].append(None)

                        
            #all hyp things
            r_dict['exp_name'].append(exp_name)
            hyps = exp_name.split('_')[2:]

            if 'h8' in exp_name:
                r_dict['h'].append(8)
            else:
                r_dict['h'].append(4)

            # r_lens = [(k,len(v)) for k,v in r_dict.items()]
            # # for l in r_lens:
            # if r_lens[-1][1] != r_lens[0][1]:
            #     print(r_lens)
            #     print('failed at exp_name {}'.format(exp_name))
            #     sys.exit(1)            
            
            if 'freeze_vae' in exp_name:
                r_dict['freeze'].append('freeze_vae')
            elif 'freeze_dec' in exp_name:
                r_dict['freeze'].append('freeze_dec')
            elif 'no_freeze' in exp_name:
                r_dict['freeze'].append('no_freeze')
            

            if "_e" in exp_name:
                r_dict['e_r'].append(100)
            else: 
                r_dict['e_r'].append(50)

            if 'dm' in exp_name:
                dm = int(exp_name.split('dm')[1].split('_')[0])
                r_dict['dm'].append(dm)
            else:
                r_dict['dm'].append(128)

            if '_k1' in exp_name:
                r_dict['k1'].append(float(exp_name.split('k1')[1].split('_')[0]))
            else: 
                r_dict['k1'].append(1.0)

            if '_lr' in exp_name:
                r_dict['lr'].append(float(exp_name.split('lr')[1].split('_')[0]))
            else: 
                r_dict['lr'].append(1e-3)

    except FileNotFoundError as e:
        print("could not find {} for exp {}".format(summary,exp_name))
        continue

try:
    r_lens = [(k,len(v)) for k,v in r_dict.items()]
    for l in r_lens:
        # print(l[0])
        # print(l[1])
        # print(r_lens[0][1])
        assert(l[1]==r_lens[0][1])
        # print(pd.DataFrame.from_dict(r_dict).head())
        # 
    pd.DataFrame.from_dict(r_dict).to_csv('results.csv',index=False)

except AssertionError:
    print("error that makes everything useless")
    sys.exit(1)            

# print(r_dict)
# pd.DataFrame(r_dict).to_csv('results.csv',index=False)

        # extract validation results
        # for r in subf:
        #     print(r)
        
        # current_file = os.path.join(subf,'validation_summary.csv')
        # df = pd.read_csv(current_file)


        # print(current_file)
