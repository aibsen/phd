import pandas as pd
import os, sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,log_loss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import *
from datasets import LCs


results_dir = "../../results/"
# exp_name = 'plasticc_cropped_gru/seed_1772670'
exp_name = 'plasticc_test_grusa_eg/seed_1772670'
where = results_dir+exp_name+'/result_outputs/'
data_dir_train = "/home/ai/phd/data/plasticc/dmdt/training/"
data_dir_csv = "/home/ai/phd/data/plasticc/csvs/"

def merge_files():
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
    all_predictions.to_csv(where+"all_test_results.csv",sep=',',index=False)

    all_probabilities = pd.concat(probabilities, axis=0, ignore_index=True)
    all_probabilities.to_csv(where+"all_test_probabilities.csv",sep=',',index=False)


def overall_summary(csv_results,csv_probabilities):
    results = pd.read_csv(where+csv_results)
    predictions = results.prediction
    probabilities = pd.read_csv(where+csv_probabilities).iloc[:,1:]

    targets = results.target
    # print(targets.values)
    f1_macro = f1_score(targets, predictions, average='weighted')
    f1_no_99 = f1_score(targets, predictions,labels=range(0,9), average='weighted') # 0.3506195828251394
    f1_sn = f1_score(targets, predictions, labels=range(0,6),average='weighted') #0.1966493728416829
    #no 99 f1: 0.354, sn only: 0.2123
    print("f1 : "+str(f1_macro))
    print("f1 without 99 : "+str(f1_no_99))
    print("f1 sn only : "+str(f1_sn)) 

    accuracy = accuracy_score(targets, predictions)
    print("accuracy : "+str(accuracy)) #no 99 0.572

    precision = precision_score(targets, predictions,average='weighted')
    recall = recall_score(targets, predictions,average='weighted')

    # weight_code = [1,1,1,1,1,1,2,2,1,1,1,1,1,1]
    weight_code = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # weight_code = [1,1,1,1,1,1,2,2,1,1,1,1,1,1,2,2,2,2]
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
    metrics.to_csv(where+"all_summary_test_micro.csv",index=False)

def overall_cm(csv_results,output_name):
    out = where+output_name
    results = pd.read_csv(where+csv_results)
    predictions = results.prediction
    targets = results.target
    # print(plasticc_names)
    plot_best_val_cm(targets,predictions,save=True, output_file=out,names=plasticc_names,normalized=False)

 

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

def how_many():
    # train_file = data_dir_train+"dmdts_training_24x24_b00_new8k.h5"
    # train_data = LCs(24,train_file,n_channels=6)
    # train_data.load_data_into_memory()
    # labels = train_data.Y
    # print(len(train_data))
    # print(set(labels.cpu().numpy()))
    train_metadata_fn = data_dir_csv+"plasticc_new8k_train_metadata.csv"
    train_metadata = pd.read_csv(train_metadata_fn)
    print(train_metadata.groupby('true_target').count())

def compare_cms(exp_names,names,classes,output_name):
    # files = [results_dir+exp_name+'/seed_1772670/result_outputs/test_1_results.csv' for exp_name in exp_names]
    files = [results_dir+exp_name+'/result_outputs/test_0.25_2_results.csv' for exp_name in exp_names]
    out = results_dir+output_name
    plot_cms(files,2,2, subtitles=names,classes=classes, 
        save=True,
        output_file=out)



# merge_files()
# overall_summary('all_test_results.csv','all_test_probabilities.csv')
overall_cm('all_test_results.csv','all_cm_real.png')

# names = ["FCN","FCN", "ResNet","ResNet", "RNN","RNN", "RNN-SA","RNN-SA"]
# names = ["FCN", "ResNet", "RNN", "RNN-SA"]
# exp_names = ["plasticc_{}".format(model) for model in ["vanilla_fcn_eg","test_fcn_eg","vanilla_resnet_eg","test_resnet_eg","vanilla_gru_eg","test_gru_eg","vanilla_grusa_eg","test_grusa_eg"]]
# exp_names = ["plasticc_balanced_cropped_{}".format(model) for model in ["fcn","resnet","gru","grusa"]]
# compare_cms(exp_names,names,plasticc_names,"plasticc_balanced_cropped_0.25_2.png")

