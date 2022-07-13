import pandas as pd
import os, sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,log_loss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import *
import torch.nn.functional as F

results_dir = "../../results/"
exp_name = '2_24x24final'
where = results_dir+exp_name+'/result_outputs/'
run_number = '2'


def soften(csv_probabilities):
    probabilities = pd.read_csv(where+csv_probabilities).iloc[:,1:]
    print(probabilities)
    probabilities = probabilities.transform('len',axis=0)
    print(probabilities)

    # probabilities.to_csv(where+"all_test_probabilities_tame2.csv".format(run_number),index=False)


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

soften('all_test_probabilities_tame.csv')