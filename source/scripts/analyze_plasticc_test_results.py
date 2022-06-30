import pandas as pd
import os, sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,log_loss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import *

results_dir = "../../results/"
exp_name = '2_24x24final'
where = results_dir+exp_name+'/result_outputs/'

def merge_files():
    predictions = []
    probabilities = []
    
    for i in range(1,12):
        predictions_fn = where+'test_{}_tame_results.csv'.format(i)
        pred = pd.read_csv(predictions_fn)
        predictions.append(pred)
        
        probabilities_fn = where+'test_{}_tame_probabilities.csv'.format(i)
        prob = pd.read_csv(probabilities_fn)
        probabilities.append(prob)
        
    all_predictions = pd.concat(predictions, axis=0, ignore_index=True)
    all_predictions.to_csv(where+"all_test_results_tame.csv",sep=',',index=False)

    all_probabilities = pd.concat(probabilities, axis=0, ignore_index=True)
    all_probabilities.to_csv(where+"all_test_probabilities_tame.csv",sep=',',index=False)


def overall_summary(csv_results,csv_probabilities):
    results = pd.read_csv(where+csv_results)
    predictions = results.prediction
    probabilities = pd.read_csv(where+csv_probabilities).iloc[:,1:]

    targets = results.target
    # print(targets.values)
    f1_macro = f1_score(targets, predictions, average='macro')
    f1_no_99 = f1_score(targets, predictions,labels=range(0,14), average='macro') # 0.3506195828251394
    f1_sn = f1_score(targets, predictions, labels=range(0,6),average='macro') #0.1966493728416829
    print("f1 : "+str(f1_macro))
    print("f1 without 99 : "+str(f1_no_99))
    print("f1 sn only : "+str(f1_sn)) 

    accuracy = accuracy_score(targets, predictions)
    print("accuracy : "+str(accuracy))

    precision = precision_score(targets, predictions,average='micro')
    recall = recall_score(targets, predictions,average='micro')

    weight_code = [1,1,1,1,1,1,2,2,1,1,1,1,1,1,2]
    weights = [weight_code[i] for i in targets.values] 
    # print(probabilities.values.shape)
    # print(targets.shape)
    loss = log_loss(targets,probabilities,sample_weight=weights) #1.65999785657598
    loss_all_equal = log_loss(targets,probabilities) #1.6206824740333845


    # loss_no_99 = log_loss(targets,probabilities,labels=range(0,14))
    
    print("loss : "+str(loss))
    print("loss balanced: "+str(loss_all_equal))
    # print("loss without 99: "+str(loss_no_99))

    metrics = pd.DataFrame({'accuracy': accuracy, 'loss':loss,"f1":f1_macro, "precision":precision, "recall": recall},index=[0])
    metrics.to_csv(where+"all_summary_dramatic.csv",index=False)

def overall_cm(csv_results,output_name):
    out = where+output_name
    results = pd.read_csv(where+csv_results)
    predictions = results.prediction
    targets = results.target
    plot_best_val_cm(targets,predictions,save=True, output_file=out,names=plasticc_names)

 

plasticc_type_dict = {
    '90':'SN-Ia',
    '67':'SN-Ia-91BG', 
    '52':'SN-Iax',  
    '42':'SN-II', 
    '62':'SN-Ib/c',
    '95':'SLSN',
    '15':'TDE',
    '64':'KN',
    '88':'AGN',
    '92':'RRL',
    '65':'M-dwarf',
    '16':'EB',
    '53':'Mira',
    '6':'uLens-Single',
    '99':'Class 99'
}
plasticc_types = [90,67,52,42,62,95,15,64,88,92,65,16,53,6,99]
plasticc_names = [plasticc_type_dict[k] for k in plasticc_type_dict]

def probs_to_plasticc_format(csv_probs):
    mapper = lambda x: x if x=='object_id' else 'class_'+str(plasticc_types[int(x)])
    probabilities = pd.read_csv(where+csv_probs)
    print(probabilities.head())
    probabilities.rename(mapper,axis=1,inplace=True)
    print(probabilities.head())
    new_order_keys = ["object_id","class_6","class_15","class_16","class_42","class_52","class_53","class_62","class_64","class_65","class_67","class_88","class_90","class_92","class_95","class_99"]
    probabilities = probabilities[new_order_keys]
    print(probabilities.head())
    probabilities.to_csv(where+'all_test_probabilities_plasticc_tame.csv',index=False)

# merge_files()
# overall_summary('all_test_results_dramatic.csv','all_test_probabilities_dramatic.csv')
# overall_cm('all_test_results_dramatic.csv','all_cm_dramatic.png')
probs_to_plasticc_format('all_test_probabilities_tame.csv')