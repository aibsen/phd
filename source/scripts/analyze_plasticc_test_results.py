import pandas as pd
import os, sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,log_loss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import *
from datasets import LCs


results_dir = "../../results/"
# exp_name = 'plasticc_cropped_gru/seed_1772670'
exp_name = 'plasticc_balanced_cropped_fcn_eg_3_p5/seed_1772670'
where = results_dir+exp_name+'/result_outputs/'
data_dir_train = "/home/ai/phd/data/plasticc/dmdt/training/"
data_dir_csv = "/home/ai/phd/data/plasticc/csvs/"


plasticc_type_dict = {
    '90':'SN-Ia',
    '67':'SN-Ia-91BG', 
    '52':'SN-Iax',  
    '42':'SN-II', 
    '62':'SN-Ib/c',
    '95':'SLSN',
    '15':'TDE',
    '64':'KN',
    '88':'AGN'
    # '92':'RRL',
    # '65':'M-dwarf',
    # '16':'EB',
    # '53':'Mira',
    # '6':'uLens-Single',
    # '99':'Class 99'
    # # '991':'uLens-Binary',
    # '992':'ILOT',
    # '993':'CART',
    # '994': 'PISN'
}
plasticc_types = [90,67,52,42,62,95,15,64,88]
#,92,65,16,53,6,99]
plasticc_names = [plasticc_type_dict[k] for k in plasticc_type_dict]

def merge_test_results_files_cv(folds=5):
    for fold in range(folds):
        print("Merging test files in fold {}".format(fold+1))
        where_in_fold = results_dir+exp_name+'/folds/fold_k{}/result_outputs/'.format(fold+1)
        merge_test_result_files(where_in_fold)

def overall_test_summary_cv(folds=5):
    for fold in range(folds):
        where_in_fold = results_dir+exp_name+'/folds/fold_k{}/result_outputs/'.format(fold+1)
        overall_test_summary(where_in_fold)

def overall_cm_cv(folds=5):
    for fold in range(folds):
        where_in_fold = results_dir+exp_name+'/folds/fold_k{}/result_outputs/'.format(fold+1)
        overall_cm(where_in_fold)

def merge_test_result_files(where):
    predictions = []
    probabilities = []
    for i in range(3,12):
        predictions_fn = where+'test_batch{}_results.csv'.format(i)
        pred = pd.read_csv(predictions_fn)
        predictions.append(pred)
        
        probabilities_fn = where+'test_batch{}_probabilities.csv'.format(i)
        prob = pd.read_csv(probabilities_fn)
        probabilities.append(prob)
        
    all_predictions = pd.concat(predictions, axis=0, ignore_index=True)
    all_predictions.to_csv(where+"test_results.csv",sep=',',index=False)

    all_probabilities = pd.concat(probabilities, axis=0, ignore_index=True)
    all_probabilities.to_csv(where+"test_probabilities.csv",sep=',',index=False)

def average_test_summaries(where):
    metrics = {}
    test_batches =range(3,12)
    for i in test_batches:
        test_summary = pd.read_csv(where+'test_batch{}_summary.csv'.format(i))
        if i == test_batches[0]:
            metrics = {k: [] for k in test_summary.keys()}
        for metric in test_summary.keys():
            metrics[metric].append(test_summary[metric].values[0])

    mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(mean_metrics)
    pd.DataFrame(mean_metrics,index=[0]).to_csv(where+"test_summary.csv",index=False)

def vote_on_test_results_cv(folds = 5, predictions_fn="test_results.csv"):
    predictions = None
    winner_predictions = []
    for fold in range(folds):
        print("Fold {}".format(fold+1))
        where_in_fold = results_dir+exp_name+'/folds/fold_k{}/result_outputs/'.format(fold+1)
        fold_predictions = pd.read_csv(where_in_fold+predictions_fn)
        if predictions is None: 
            predictions = np.zeros((fold_predictions.shape[0],6))
        for idx,row in fold_predictions.iterrows():
            prediction_idx = row.prediction
            predictions[idx,prediction_idx]+=1

    winner = np.argmax(predictions,axis=1)
    fold_predictions.loc[:,'prediction']=winner
    fold_predictions.to_csv(results_dir+exp_name+'/result_outputs/majority_test_results.csv')

def overall_test_summary(where,csv_results="test_results.csv",csv_probabilities="test_probabilities.csv",output="test_summary.csv"):
    results = pd.read_csv(where+csv_results)
    predictions = results.prediction
    probabilities = pd.read_csv(where+csv_probabilities).iloc[:,1:]

    targets = results.target
    f1_macro = f1_score(targets, predictions, average='macro')
    f1_weighted = f1_score(targets, predictions, average='weighted')
    accuracy = accuracy_score(targets, predictions)
    print("accuracy : "+str(accuracy)) #no 99 0.572
    print("f1_macro : "+str(f1_macro)) #no 99 0.572

    precision_macro = precision_score(targets, predictions,average='macro')
    precision_weighted = precision_score(targets, predictions,average='weighted')
    recall_macro = recall_score(targets, predictions,average='macro')
    recall_weighted = recall_score(targets, predictions,average='weighted')

    weight_code = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # weight_code = [1,1,1,1,1,1,2,2,1,1,1,1,1,1,2,2,2,2]
    weights = [weight_code[i] for i in targets.values] 
    loss = log_loss(targets,probabilities,sample_weight=weights)
    print("loss : "+str(loss))

    metrics = pd.DataFrame({'accuracy': accuracy, 'loss':loss,
        "f1":f1_macro, "f1_weighted":f1_weighted, 
        "precision":precision_macro, "precision_weighted":precision_weighted, 
        "recall": recall_macro, "recall_weighted":recall_weighted}
        ,index=[0])
    metrics.to_csv(where+output,index=False)

def overall_cm(where,csv_results="test_results.csv",output_name="test_cm.png"):
    out = where+output_name
    results = pd.read_csv(where+csv_results)
    predictions = results.prediction
    targets = results.target
    cm = plot_cm(targets,predictions,save=True, output_file=out,names=plasticc_names,normalized=True)
    return cm
# def compare_cms(exp_names,names,classes,output_name):
#     # files = [results_dir+exp_name+'/seed_1772670/result_outputs/test_1_results.csv' for exp_name in exp_names]
#     files = [results_dir+exp_name+'/result_outputs/test_0.25_2_results.csv' for exp_name in exp_names]
#     out = results_dir+output_name
#     plot_cms(files,2,2, subtitles=names,classes=classes, 
#         save=True,
#         output_file=out)


# average_test_summaries(results_dir+'plasticc_test_resnet_sn/result_outputs/')
# overall_test_summary(where,csv_results="majority_test_results.csv",)
merge_test_results_files_cv()
overall_test_summary_cv()
overall_cm_cv()
# vote_on_test_results_cv()
# overall_cm(where,csv_results="majority_test_results.csv")


# names = ["FCN","FCN", "ResNet","ResNet", "RNN","RNN", "RNN-SA","RNN-SA"]
# names = ["FCN", "ResNet", "RNN", "RNN-SA"]
# exp_names = ["plasticc_{}".format(model) for model in ["vanilla_fcn_eg","test_fcn_eg","vanilla_resnet_eg","test_resnet_eg","vanilla_gru_eg","test_gru_eg","vanilla_grusa_eg","test_grusa_eg"]]
# exp_names = ["plasticc_balanced_cropped_{}".format(model) for model in ["fcn","resnet","gru","grusa"]]
# compare_cms(exp_names,names,plasticc_names,"plasticc_balanced_cropped_0.25_2.png")

