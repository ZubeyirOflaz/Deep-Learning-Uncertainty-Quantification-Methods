import torch
from laplace import Laplace
from torch.utils.data import DataLoader
import torch.distributions as dists
from netcal.metrics import ECE
import numpy

from torch.utils.data import Dataset, TensorDataset
import pickle
# import helper
from config import dataset_paths, models
import os
from laplace.utils import ModuleNameSubnetMask


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device='cuda')
            y = y.to(device='cuda')
            y = torch.reshape(y, (-1,))

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()



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

subnetwork_mask = ModuleNameSubnetMask(arrhythmia_model, module_names=['6', '9'])
subnetwork_mask.select()
subnetwork_indices = subnetwork_mask.indices

subnetwork_indices = torch.LongTensor(subnetwork_indices.cpu())

la = Laplace(arrhythmia_model, likelihood='classification', subset_of_weights='subnetwork',
             hessian_structure='full', subnetwork_indices=subnetwork_indices)
la.fit(train_loader)
la.optimize_prior_precision(method='CV', val_loader=train_loader)

pred = predict(test_loader, la, laplace=True)

probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
# ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(f'[Laplace] Acc.: {acc_laplace:.1%} NLL: {nll_laplace:.3}')

