import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import *

results_dir = "../../results/"
exp_name = "unbalanced_dataset_m_realzp_328"

test_results = pd.read_csv(results_dir+exp_name+"/result_outputs/validation_results.csv")
true_tags = test_results.true_tags.values
predicted = test_results.predicted_tags.values
plot_cm(true_tags,predicted)
plot_cm(true_tags,predicted,False)
