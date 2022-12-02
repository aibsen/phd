import pandas as pd
import os, sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,log_loss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils
from datasets import LCs


results_dir = "../../results/"

type_dict = {
    '0':'SN-Ia',
    '1':'SN-II', 
    '2':'SN-Ib/c',
    '3':'SLSN',
}
types = [0,1,2,3]
names = [type_dict[k] for k in type_dict]

def plot_best_val_cm_cv(exp_name, folds=5,val_name=''):
    
    best_f1 = 0
    where = None

    for fold in range(folds):
        where_in_fold = exp_name+'/folds/fold_k{}/result_outputs/'.format(fold+1)
        out = where_in_fold+'{}_validation_summary.csv'.format(val_name)
        info = pd.read_csv(out)
        max_f1 = info.f1.max()
        if max_f1 > best_f1:
            best_f1 = max_f1
            where = where_in_fold

    print("plotting best fold val cm...")

    plot_cm(where, csv_results="{}_validation_results.csv".format(val_name), output_fname=exp_name+"/result_outputs/{}_best_validation_cm.png".format(val_name))


def plot_all_cm_cv(folds=5, results_dir=results_dir):
    for fold in range(folds):
        where_in_fold = results_dir+'/folds/fold_k{}/result_outputs/'.format(fold+1)
        plot_cm(where_in_fold, csv_results="validation_results.csv", output_fname=where_in_fold+"valdiation_cm.png")

def plot_cm(where,csv_results="test_results.csv",output_fname="test_cm.png"):
    results = pd.read_csv(where+csv_results)
    predictions = results.prediction
    targets = results.target
    cm = plot_utils.plot_cm(targets,predictions,save=True, output_file=output_fname,names=names,normalized=True)
    return cm

# data_reps = ['linear', 'gp', 'uneven_tnorm_backl']
# codes=[3,4,5,6,7,8,9]
# for exp_code in codes:
#     for rep in data_reps:
#         exp_name = results_dir+'data_rep_exp_{}_{}/seed_1772670'.format(exp_code,rep)
#         plot_all_cm_cv(folds=5,results_dir=exp_name)
#         plot_cm(exp_name+'/result_outputs/',"test_results.csv",exp_name+'/result_outputs/test_cm.png')
#         plot_best_val_cm_cv(exp_name)


# def compare_cms(exp_names,names,classes,output_name):
#     files = [results_dir+exp_name+'/seed_1772670/result_outputs/all_test_results.csv' for exp_name in exp_names]
#     # files = [results_dir+exp_name+'/result_outputs/test_0.25_2_results.csv' for exp_name in exp_names]
#     out = results_dir+output_name
#     plot_cms(files,2,2, subtitles=names,classes=classes, 
#         save=True,
#         output_file=out)

# def compare_cms_folds(exp_names,names,classes,output_name):
#     files = [results_dir+exp_name+'/seed_1772670/folds/fold_k{}/result_outputs/test_results_0.25.csv'.format(i) for exp_name,i in exp_names]
#     out = results_dir+output_name
#     plot_cms(files,2,2, subtitles=names,classes=classes, 
#         save=True,
#         output_file=out)

# exp_names = [('plasticc_balanced_fcn_eg',2),('plasticc_balanced_resnet_eg',1),('plasticc_balanced_gru_eg_0',1),('plasticc_balanced_grusa_eg_0',5)]
# names = ["FCN", "ResNet", "RNN", "RNN-SA"]
# compare_cms_folds(exp_names,names, plasticc_names,'plasticc_balanced_2.png')

# exp_names = [('plasticc_balanced_fcn_eg',2),('plasticc_balanced_resnet_eg',1),('plasticc_balanced_gru_eg',1),('plasticc_balanced_grusa_eg',4)]
# names = ["FCN", "ResNet", "RNN", "RNN-SA"]
# compare_cms_folds(exp_names,names, plasticc_names,'plasticc_balanced_0.5_2.png')

# exp_names = [('plasticc_balanced_fcn_eg',4),('plasticc_balanced_resnet_eg',1),('plasticc_balanced_gru_eg',1),('plasticc_balanced_grusa_eg',4)]
# names = ["FCN", "ResNet", "RNN", "RNN-SA"]
# compare_cms_folds(exp_names,names, plasticc_names,'plasticc_balanced_0.25_2.png')

# 5,1,10,50
# average_test_summaries(results_dir+'plasticc_test_resnet_sn/result_outputs/')
# overall_test_summary(where,csv_results="majority_test_results.csv",)
# for model in ['fcn','resnet','gru', 'grusa']:
    # merge_test_result_files(results_dir+'plasticc_balanced_{}_eg/result_outputs/'.format(model))
# overall_test_summary_cv()
# overall_cm_cv()
# vote_on_test_results_cv()
# overall_cm(where,csv_results="majority_test_results.csv")


# # names = ["FCN","FCN", "ResNet","ResNet", "RNN","RNN", "RNN-SA","RNN-SA"]
# names = ["FCN", "ResNet", "RNN", "RNN-SA"]
# exp_names = ["plasticc_{}".format(model) for model in ["test_fcn_eg","test_resnet_eg","test_gru_eg","test_grusa_eg"]]
# # "test_fcn_eg","vanilla_resnet_eg","test_resnet_eg","vanilla_gru_eg","test_gru_eg","vanilla_grusa_eg","test_grusa_eg"]]
# # exp_names = ["plasticc_balanced_cropped_{}".format(model) for model in ["fcn","resnet","gru","grusa"]]
# compare_cms(exp_names,names,plasticc_names,"plasticc_test_eg_2.png")

# names = ["FCN", "ResNet", "RNN", "RNN-SA"]
# exp_names = ["plasticc_{}".format(model) for model in ["vanilla_fcn_eg","vanilla_resnet_eg","vanilla_gru_eg","vanilla_grusa_eg"]]
# # "test_fcn_eg","vanilla_resnet_eg","test_resnet_eg","vanilla_gru_eg","test_gru_eg","vanilla_grusa_eg","test_grusa_eg"]]
# # exp_names = ["plasticc_balanced_cropped_{}".format(model) for model in ["fcn","resnet","gru","grusa"]]
# compare_cms(exp_names,names,plasticc_names,"plasticc_vanilla_eg_2.png")
