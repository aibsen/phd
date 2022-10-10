import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import LCs
from seq2seq_experiment import Experiment
from transformer import TSTransformer
from recurrent_models import GRU1D
import torch

results_dir = "../../results/"
data_dir = "/home/ai/phd/data/ztf/training/linearly_interpolated/"
# data_dir = "/home/ai/phd/data/ztf/testing/"
exp_name = "transformer_trial_0"


lc_length = 128
batch_size = 64
num_epochs = 100
use_gpu = True
lr = 1e-03
wdc = 1e-03
seeds = [1772670]
n_seeds = 1

exp_params={
    "num_epochs" : num_epochs,
    "learning_rate" : lr,
    "weight_decay_coefficient" : wdc,
    "use_gpu" : use_gpu,
    "batch_size" : batch_size,
    "chunked": False,
    "num_output_classes": 14
}


# training_data_file=data_dir+'real_data_careful.h5'
training_data_file=data_dir+'simsurvey_data_balanced.h5'
train_dataset = LCs(lc_length, training_data_file)
train_dataset.load_data_into_memory()
input_shape = train_dataset[0][0].shape
train_dataset.lens = torch.full((len(train_dataset),),lc_length)
train_dataset.packed = True

# nn = TSTransformer()
grusa_params = {
    "num_output_classes" : 6,
    "hidden_size":100,
    "batch_size":batch_size,
    "attention":"self_attention",
    "da":50,
    "r":1,
    'input_shape': input_shape
    }

nn=GRU1D(grusa_params)
nn=TSTransformer(nhead=4,nlayers=1)

experiment = Experiment(
    nn,
    experiment_name=exp_name,
    num_epochs=num_epochs,
    train_data=train_dataset,
    val_data=train_dataset #trivial case to check if things work
)

experiment.run_experiment()
# nn.to(torch.device('cuda'))
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# for i,(x, y, ids) in enumerate(train_loader):
# #     # print(x)
#     print(x[0].shape)
#     src = x[0].permute(0,2,1)
#     src = (src,x[1])
#     a =nn(src,src)
# #     print(a.shape)
#     break
# # for i, epoch_idx in enumerate(range(num_epochs)):
# #     print(i)
# #     break





