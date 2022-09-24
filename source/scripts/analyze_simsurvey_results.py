from termios import CSTOPB
import pandas as pd
import os, sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,log_loss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import *
# from utils import simsurvey_ztf_type_dict

results_dir = "../../results/"
exp_name = 'simsurvey_sa_r1_da20'

simsurvey_ztf_type_dict = {
    '0':'Ia',
    '1':'Ia-91bg', 
    '2':'Iax',  
    '3':'II', 
    '4':'Ibc',
    '5':'SLSN',
    # '6':'IIn'
}

ztf_names = [simsurvey_ztf_type_dict[k] for k in simsurvey_ztf_type_dict]
# print(ztf_names)

def overall_cm(where,csv_results="test_0.25_results.csv",output_name="test_0.25_cm.png"):
    out = where+output_name
    results = pd.read_csv(where+csv_results)
    predictions = results.prediction
    targets = results.target
    cm = plot_cm(targets,predictions,save=True, output_file=out,names=ztf_names,normalized=True)
    return cm

def overall_cm_cv(where_in_fold=None,folds=5,seed=1772670):
    print(where_in_fold)
    for fold in range(folds):
        if where_in_fold is None:
            where_in_fold = results_dir+exp_name+'/seed_{}/folds/fold_k{}/result_outputs/'.format(seed,fold+1)
            overall_cm(where_in_fold)
        else:
            new=where_in_fold+'/seed_{}/folds/fold_k{}/result_outputs/'.format(seed,fold+1)
            overall_cm(new)        



def overall_cm_selective(where,csv_results="test",output_name="test_selective_cm_m05.png", th=0.5):
    out = where+output_name
    probabilities = pd.read_csv(where+csv_results+"_probabilities.csv")
    results = pd.read_csv(where+csv_results+"_results.csv")

    # probabilities = probabilities[]
    max_probs_mask = probabilities[probabilities.keys()[1:]].max(axis=1) < th
    selective = probabilities[max_probs_mask].object_id
    results = results[results.object_id.isin(selective)]
    predictions = results.prediction
    targets = results.target
    cm = plot_cm(targets,predictions,save=True, output_file=out,names=ztf_names,normalized=False)
    return cm

def overall_cm_selective_cv(folds=5):
    for fold in range(folds):
        where_in_fold = results_dir+exp_name+'/folds/fold_k{}/result_outputs/'.format(fold+1)
        overall_cm_selective(where_in_fold)

def compare_cms(exp_names,names,classes,output_name):
    files = [results_dir+exp_name+'/seed_1772670/folds/fold_k{}/result_outputs/test_results.csv'.format(i) for exp_name,i in exp_names]
    out = results_dir+output_name
    plot_cms(files,2,2, subtitles=names,classes=classes, 
        save=True,
        output_file=out)

def compare_cms2(exp_names,names,classes,output_name,rows=2,cols=2):
    files = [results_dir+exp_name+'/seed_1772670/folds/fold_k{}/result_outputs/{}'.format(i,f) for (exp_name,i,f) in exp_names]
    out = results_dir+output_name
    plot_cms(files,rows,cols, subtitles=names,classes=classes, 
        save=True,
        output_file=out)

# balanced_real_2
# exp_names = [('simsurvey_balanced_1fcn',3,'test_real_results.csv'),('simsurvey_balanced_1resnet',3,'test_real_results.csv'),
#     ('simsurvey_balanced_gru',1,'test_results.csv'),('simsurvey_sa_balanced_shufflesplit_r1_da20',2,"test_old_results.csv")]
# balanced_1_test_2
exp_names = [('simsurvey_balanced_1fcn',3),('simsurvey_balanced_1resnet',1),
    ('simsurvey_balanced_1gru',1),('simsurvey_balanced_1grusa',3)]

#cropped 1 sa_0.5 = r5,da50; sa_0.25= r3,da60
# exp_names = [('simsurvey_cropped_fcn',5),('simsurvey_cropped_resnet',5),
#     ('simsurvey_cropped_fixgru',2),('simsurvey_sa_r5_da50',1)]
# cropped_real_2
# exp_names = [('simsurvey_cropped_fcn',5,'test_old_results.csv'),('simsurvey_cropped_resnet',5,'test_old_results.csv'),
#     ('simsurvey_cropped_fixgru',4,'test_real_results.csv'),('simsurvey_sa_r5_da60',1,"test_old_results.csv")]

# names = ["FCN", "ResNet", "RNN", "RNN-SA"]
# compare_cms(exp_names,names, ztf_names,'simsurvey_balanced_test_cm_2.png')

# percents
# # fcn
# exp_names = [('simsurvey_cropped_fcn',5,'test_results.csv'),('simsurvey_cropped_fcn',5,'test_0.5_results.csv'),
#     ('simsurvey_cropped_fcn',5,'test_0.25_results.csv')]
# #resnet
# exp_names = [('simsurvey_cropped_resnet',2,'test_results.csv'),('simsurvey_cropped_resnet',2,'test_0.5_results.csv'),
#     ('simsurvey_cropped_resnet',2,'test_0.25_results.csv')]    
# # # gru
# exp_names = [('simsurvey_cropped_fixgru',2,'test_results.csv'),('simsurvey_cropped_fixgru',2,'test_0.5_results.csv'),
#     ('simsurvey_cropped_fixgru',2,'test_0.25_results.csv')]
#grusa
# exp_names = [('simsurvey_sa_r5_da50',1,'test_results.csv'),('simsurvey_sa_r5_da50',1,'test_0.5_results.csv'),
#     ('simsurvey_sa_r3_da60',1,'test_0.25_results.csv')]
# names = ["100%", "50%", "25%"]
# compare_cms2(exp_names,names, ztf_names,'simsurvey_fcn_cropped.png',rows=1,cols=3)

# names = ["100%", "50%", "25%"]
# # ,"100%", "50%", "25%","100%", "50%", "25%","100%", "50%", "25%"]
# #all percents
# exp_names = [
#     ('simsurvey_sa_r6_da5',2,'test_results.csv'),('simsurvey_sa_r6_da5',2,'test_0.5_results.csv'),
#     ('simsurvey_sa_r6_da5',2,'test_0.25_results.csv'),
#     ('simsurvey_cropped_fixgru',2,'test_results.csv'),('simsurvey_cropped_fixgru',2,'test_0.5_results.csv'),
#     ('simsurvey_cropped_fixgru',2,'test_0.25_results.csv'),
#     ('simsurvey_cropped_fcn',5,'test_results.csv'),('simsurvey_cropped_fcn',5,'test_0.5_results.csv'),
#     ('simsurvey_cropped_fcn',5,'test_0.25_results.csv'),
#     ('simsurvey_cropped_resnet',1,'test_results.csv'),('simsurvey_cropped_resnet',1,'test_0.5_results.csv'),
#     ('simsurvey_cropped_resnet',1,'test_0.25_results.csv')]

# compare_cms2(exp_names,names, ztf_names,'simsurvey_cropped_all.png',rows=4,cols=3)




# # # overall_cm_selective_cv()
# overall_cm_cv()
# def clean_test_files(folds=5,seed=1772670,template='test_real_lasairmars5pb'):
#     seed_overall = results_dir+exp_name+'/result_outputs/{}_summary.csv'.format(template)
#     fold_overall = results_dir+exp_name+'/seed_{}/result_outputs/{}_summary.csv'.format(seed,template)
#     os.remove(seed_overall)
#     os.remove(fold_overall)
#     for fold in range(folds):
#         where_in_fold = results_dir+exp_name+'/seed_{}/folds/fold_k{}/result_outputs/'.format(seed,fold+1)
#         os.remove(where_in_fold+template+'_results.csv')
#         os.remove(where_in_fold+template+'_probabilities.csv')
#         os.remove(where_in_fold+template+'_summary.csv')

# clean_test_files()