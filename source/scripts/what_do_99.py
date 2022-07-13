import pandas as pd
import os, sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,log_loss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import *
import matplotlib.pyplot as plt
results_dir = "../../results/"
exp_name = '2_24x24final'
where = results_dir+exp_name+'/result_outputs/'


def inspect_99(csv_results,csv_probabilities):
    results = pd.read_csv(where+csv_results)
    probabilities = pd.read_csv(where+csv_probabilities)

    cond = probabilities.iloc[:,1:] > 0.85
    where_is_mask = probabilities.where(cond).any(axis=1)
    # print(where_is_mask)
    # print(probabilities[~where_is_mask].object_id)
    maybe_99_ids = probabilities[~where_is_mask].object_id
    maybe_99 = results[results.object_id.isin(maybe_99_ids)]
    # print(maybe_99.groupby('target').count())
    maybe_99_probs = probabilities[~where_is_mask]
    # print(maybe_99_probs)
    
    maybe_99_probs = maybe_99_probs[(np.abs(maybe_99_probs['0']-maybe_99_probs['3'])<0.1) & (np.abs(maybe_99_probs['3']-maybe_99_probs['10'])<0.1) & (np.abs(maybe_99_probs['10']-maybe_99_probs['11'])<0.05)] #confused about main classes
    # # print(maybe_99_probs)
    cond2 = maybe_99_probs.iloc[:,2:3] > 0.1#neither Ia91 or Iax is prominent
    print(cond2)
    print(maybe_99_probs)
    where_is_mask2 = maybe_99_probs.where(cond2).any(axis=1)
    print(where_is_mask2)
    maybe_99_probs = maybe_99_probs[~where_is_mask2] # none of the other explosive classes is particularly prominent
    print(maybe_99_probs)

    cond2 = maybe_99_probs.iloc[:,5:6] > 0.05 #neither Ib/c or SLSN is prominent
    print(cond2)
    print(maybe_99_probs)
    where_is_mask2 = maybe_99_probs.where(cond2).any(axis=1)
    print(where_is_mask2)
    maybe_99_probs = maybe_99_probs[~where_is_mask2] # none of the other explosive classes is particularly prominent
    print(maybe_99_probs)

    cond2 = maybe_99_probs.iloc[:,8] > 0.1 #TDE is prominent
    print(cond2)
    print(maybe_99_probs)
    where_is_mask2 = maybe_99_probs.where(cond2).any(axis=1)
    print(where_is_mask2)
    maybe_99_probs = maybe_99_probs[~where_is_mask2] # none of the other explosive classes is particularly prominent
    print(maybe_99_probs)

    maybe_99_ids =maybe_99_probs.object_id
    maybe_99 = results[results.object_id.isin(maybe_99_ids)]
    print(maybe_99.groupby('target').count())
    print(maybe_99.groupby('target').count().sum())
    print(maybe_99.groupby('target').count().object_id[14])
    print(maybe_99.groupby('target').count().object_id[14]/maybe_99.groupby('target').count().sum())

    print(maybe_99_ids)

    probabilities.loc[probabilities.object_id.isin(maybe_99_ids),['0','3']] = 0.2
    probabilities.loc[probabilities.object_id.isin(maybe_99_ids),'14'] = 0.3
    probabilities.loc[probabilities.object_id.isin(maybe_99_ids),'10'] = 0.1
    probabilities.loc[probabilities.object_id.isin(maybe_99_ids),['1','2','4','5','7']] = 0.4/5
    probabilities.loc[probabilities.object_id.isin(maybe_99_ids),['6','8','9','11','12','13']] = 0.0

    results.loc[results.object_id.isin(maybe_99_ids),"prediction"] = 14
    probabilities.to_csv(where+'all_test_probabilities_2_14.csv',index=False)
    results.to_csv(where+'all_test_results_2_14.csv',index=False)

    print(probabilities[probabilities.object_id.isin(maybe_99_ids)])

inspect_99('all_test_results_2.csv','all_test_probabilities_2.csv')
