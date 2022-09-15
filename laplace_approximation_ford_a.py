import os
from config import dataset_paths as args
from config import models as models
import torch
import torch.nn.functional as F
from utils.ford_a_dataloader import ARFFDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time
import logging
import pandas as pd
import pickle
from laplace import Laplace
from laplace.utils import ModuleNameSubnetMask
import torch.distributions as dists
from netcal.metrics import ECE
from utils.evaluation_metrics import predict, calculate_metric_laplace,\
    create_metric_dataframe, get_metrics, benchmark_laplace



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['ford_a_train']
test_set_path = ROOT_DIR + args['ford_a_test']
model_path = ROOT_DIR + models['base_models']['ford_a']

batch_size = 36
num_workers = 0

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def ford_a_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.tensor(target).type(torch.LongTensor)
    data = torch.stack(data).unsqueeze(1)
    return [data, target]

train_dataset = ARFFDataset(train_set_path, data_scaling = True)
test_dataset = ARFFDataset(test_set_path, data_scaling = True)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=num_workers, pin_memory= True, collate_fn= ford_a_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         num_workers=num_workers, pin_memory= True, collate_fn= ford_a_collate, shuffle=False)

with open(model_path, "rb") as fin:
    ford_a_model = pickle.load(fin)

ford_a_model =  ford_a_model.to(device)
print(ford_a_model)

targets = torch.cat([y for x, y in test_loader], dim=0).cpu()
targets = torch.reshape(targets,(-1,)).to(device)
targets_train = torch.cat([y for x, y in train_loader], dim=0).cpu()
targets_train = torch.reshape(targets_train,(-1,)).to(device)


probs_map = predict(test_loader, ford_a_model).to(device)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
print(f'[MAP] Acc.: {acc_map:.1%}; NLL: {nll_map:.3}')
get_metrics(probs_map,targets)
probs_map_train = predict(train_loader,ford_a_model).to(device)
print('Metrics for train dataset')
get_metrics(probs_map_train,targets_train)

subnetwork_mask = ModuleNameSubnetMask(ford_a_model, module_names=['29'])
subnetwork_mask.select()
subnetwork_indices = subnetwork_mask.indices

subnetwork_indices = torch.LongTensor(subnetwork_indices.cpu())

la = Laplace(ford_a_model, likelihood='classification', subset_of_weights='subnetwork',
             hessian_structure='full', subnetwork_indices=subnetwork_indices)
print('fitting started')
la.fit(train_loader)
print('fitting complete')
la.optimize_prior_precision(method='CV', val_loader=test_loader, pred_type='nn', verbose=True)





pred = predict(test_loader, la, laplace=True)

probs_laplace = predict(test_loader, la, laplace=True).to(device)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
#ece_laplace = ECE(bins=10).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()
get_metrics(probs_laplace,targets,10, n_class = 2)
print(f'[Laplace] Acc.: {acc_laplace:.1%} NLL: {nll_laplace:.3}')

results_dict = calculate_metric_laplace(la,test_loader,100)
ford_a_test_dataframe = create_metric_dataframe(results_dict,mimo_metric=False)
ford_a_test_dataframe['dataset'] = 'test'
results_dict = calculate_metric_laplace(la,train_loader,100)
ford_a_train_dataframe = create_metric_dataframe(results_dict,mimo_metric=False)
ford_a_train_dataframe['dataset'] = 'train'
ford_a_dataframe = pd.concat([ford_a_train_dataframe,ford_a_test_dataframe], ignore_index= True)