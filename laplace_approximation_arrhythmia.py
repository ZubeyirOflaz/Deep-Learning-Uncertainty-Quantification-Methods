import torch
from laplace import Laplace
from torch.utils.data import DataLoader
import torch.distributions as dists
from netcal.metrics import ECE
import numpy

from torch.utils.data import Dataset, TensorDataset
import pickle
from utils.evaluation_metrics import predict, calculate_metric_laplace
from config import dataset_paths, models
import os
from laplace.utils import ModuleNameSubnetMask





ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
arrhythmia_train_path = ROOT_DIR + dataset_paths['arrhythmia_train']
arrhythmia_test_path = ROOT_DIR + dataset_paths['arrhythmia_test']

arrhythmia_model_path = models['base_models']['arrhythmia']

with open(arrhythmia_model_path, "rb") as fin:
    arrhythmia_model = pickle.load(fin)

with open(arrhythmia_train_path, 'rb') as fin:
    arrhythmia_train = numpy.load(fin, allow_pickle=True)

with open(arrhythmia_test_path, 'rb') as fin:
    arrhythmia_test = numpy.load(fin, allow_pickle=True)

#Loading and preprocessing datasets, inputing some of the hyperparameters
batch_size = 16
num_workers = 0

params = {'batch_size': batch_size,
          'num_workers': num_workers}

train_x = arrhythmia_train[:, 0]
train_y = arrhythmia_train[:, 1]

train_x = torch.tensor([i.numpy() for i in train_x]).to(torch.device('cuda'))
train_y = torch.tensor([i.numpy() for i in train_y]).to(torch.device('cuda')).flatten()
train = TensorDataset(train_x, train_y)
train_loader = DataLoader(train, shuffle=False, **params)

test_x = arrhythmia_test[:,0]
test_y = arrhythmia_test[:,1]

test_x = torch.tensor([i.numpy() for i in test_x]).to(torch.device('cuda'))
test_y = torch.tensor([i.numpy() for i in test_y]).to(torch.device('cuda')).flatten()
test = TensorDataset(test_x, test_y)

test_loader = DataLoader(test, shuffle=False, **params)

arrhythmia_model = arrhythmia_model.to(torch.device('cuda'))
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()
targets = torch.reshape(targets,(-1,))

probs_map = predict(test_loader, arrhythmia_model)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
print(f'[MAP] Acc.: {acc_map:.1%}; NLL: {nll_map:.3}')

torch.cuda.empty_cache()

subnetwork_mask = ModuleNameSubnetMask(arrhythmia_model, module_names=['9'])
subnetwork_mask.select()
subnetwork_indices = subnetwork_mask.indices

subnetwork_indices = torch.LongTensor(subnetwork_indices.cpu())

la = Laplace(arrhythmia_model, likelihood='classification', subset_of_weights='subnetwork',
             hessian_structure='full', subnetwork_indices=subnetwork_indices)
la.fit(train_loader)
la.optimize_prior_precision(method='CV', val_loader=train_loader, n_samples=250, cv_loss_with_var= True, n_steps=200, lr= 5e-3)

pred = predict(test_loader, la, laplace=True)

probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(f'[Laplace] Acc.: {acc_laplace:.1%} NLL: {nll_laplace:.3}')

