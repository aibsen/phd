import numpy as np 
import pandas as pd 
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeded_experiment import SeededExperiment
from sklearn import metrics


results_dir = "../../results/"
exp_name = "exp2_p1_fcn"
exp=2
part =1
n_folds = 5
exp_names = list(map(lambda x: x.format(exp,part),["exp{}_p{}_fcn", "exp{}_p{}_resnet", "exp{}_p{}_gru", "exp{}_p{}_grusa"]))
test_file = "test_results_new_count25.csv"


def how_many_well_predicted(test_results):
    right = test_results[test_results["predicted_tags"]==test_results["true_tags"]]
    wrong = test_results[test_results["predicted_tags"]!=test_results["true_tags"]]
    return right,wrong

def how_many_well_predicted_per_class(test_results, right, wrong):
    classes = np.sort(test_results["true_tags"].unique())
    right_per_class = right.groupby(["true_tags"]).count()
    wrong_per_class = wrong.groupby(["true_tags"]).count()



    joint = right_per_class.join(wrong_per_class,lsuffix='_right', rsuffix='_wrong',how="outer").fillna(0)
    joint = joint[["predicted_tags_right", "predicted_tags_wrong"]]

    data = {
        "class" : classes,
        "right" : joint["predicted_tags_right"].values,
        "wrong" : joint["predicted_tags_wrong"].values
    }
    per_class=pd.DataFrame(data)
    per_class["total"]=per_class["right"]+per_class["wrong"]
    per_class["right_percentage"] = 100*per_class["right"]/per_class["total"]
    per_class["wrong_percentage"] = 100*per_class["wrong"]/per_class["total"]
    return per_class

# for i, exp in enumerate(exp_names):
#     se = SeededExperiment(results_dir+exp)
#     seeds = se.get_seeds_from_folders()
#     print("--------------------------------------------------------------------------------------- ")
#     print(exp)
#     rights = []
#     wrongs = []
#     count = 0
#     best_seed = 0
#     best_k = 0
#     for seed in seeds:
#         print(" ")
#         print(seed)
#         for fold in np.arange(1,n_folds+1):
#             print(" ")
#             print(fold)
#             f = results_dir+exp+"/seed_"+str(seed)+"/folds/fold_k"+str(fold)+"/result_outputs/"+test_file
#             test_results = pd.read_csv(f)
#             total = test_results.shape[0] 
#             print("total_objects :  "+str(total))
#             right, wrong = how_many_well_predicted(test_results)
#             r = right.shape[0]
#             w = wrong.shape[0]
#             rights.append(r)
#             wrongs.append(w)
#             count+=1
#             print("correctly classified : "+str(r)+" , "+str(100*r/total)+"%")
#             print("incorrectly classified : "+str(w)+" , "+str(100*w/total)+"%")
#             per_class = how_many_well_predicted_per_class(test_results, right, wrong)
#             print(per_class)
#     print(exp+" best case : "+str(np.max(rights))+"/"+str(total)+" = "+str(100*np.max(rights)/total)+"%")
#     print(exp+" avg case : "+str(np.mean(rights))+"/"+str(total)+" +/- "+str(np.std(rights)) +" = "+str(100*np.mean(rights)/total)+"%")

n_classes=4
np.set_printoptions(precision=3)

def get_f1_per_class():
    for i, exp in enumerate(exp_names):
        se = SeededExperiment(results_dir+exp)
        seeds = se.get_seeds_from_folders() 
        n_seeds = len(seeds)
        print(exp)
        seed_mean=np.zeros((n_seeds,n_classes))
        for j,seed in enumerate(seeds):

            f = np.ndarray((n_seeds,n_folds,n_classes))            
            fold_mean = np.zeros((n_folds,n_classes))  

            for fold in np.arange(1,n_folds+1):
                filename = results_dir+exp+"/seed_"+str(seed)+"/folds/fold_k"+str(fold)+"/result_outputs/"+test_file
                test_results = pd.read_csv(filename)
                y_true = test_results["true_tags"]
                y_pred = test_results["predicted_tags"]   
                classes = test_results["true_tags"].unique()

                for c in classes:
                    y_true_c = y_true.copy()
                    y_pred_c = y_pred.copy()
                    y_true_c.loc[y_true_c!=c] = len(classes)
                    y_pred_c.loc[y_pred_c!=c] = len(classes)
                    p,r,f1,s=metrics.precision_recall_fscore_support(y_true_c, y_pred_c, pos_label =c, average="binary")
                    fold_mean[fold-1,c] = f1
                    # if f1< 0:
                    # print(f1)           
            seed_mean[j] = np.mean(fold_mean,axis=0)
        # print(seed_mean)
        # print(seed_mean.shape)
        f1_per_class = np.mean(seed_mean,axis=(0))
        print(f1_per_class)
get_f1_per_class()