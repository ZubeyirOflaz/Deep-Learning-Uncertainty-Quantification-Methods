import laplace
import torch
from laplace import Laplace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributions as dists
from netcal.metrics import ECE

from torch.utils.data import Dataset, random_split
import pickle
# import helper
import logging
from config import dataset_paths, models, casting_args
import os
from laplace.utils import LargestMagnitudeSubnetMask, ModuleNameSubnetMask

def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()



# Laplace approximation for casting dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

casting_train_path = ROOT_DIR + dataset_paths['casting_train']
casting_test_path = ROOT_DIR + dataset_paths['casting_test']

casting_model_path = models['base_models']['casting']

with open(casting_model_path, "rb") as fin:
    casting_model = pickle.load(fin)

batch_size = 32
image_resolution = 127
num_workers = 0

transformations = transforms.Compose([transforms.Resize(int((image_resolution + 1) / 2) * 3),
                                      transforms.RandomCrop(image_resolution),
                                      transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

train_set = datasets.ImageFolder(casting_train_path, transform=transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True
                                           , num_workers=num_workers, pin_memory=True)

test_set = datasets.ImageFolder(casting_test_path, transform=transformations)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

probs_map = predict(test_loader, casting_model, laplace=False)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()


torch.cuda.empty_cache()
subnetwork_mask = LargestMagnitudeSubnetMask(casting_model, n_params_subnet=64)
subnetwork_indices = subnetwork_mask.select()

subnetwork_indices = torch.LongTensor(subnetwork_indices.cpu())
#subnetwork_mask = ModuleNameSubnetMask(casting_model, module_names=['layer.9', 'layer.12'])
#subnetwork_mask.select()
#subnetwork_indices = subnetwork_mask.indices


la = Laplace(casting_model, likelihood='classification', subset_of_weights='subnetwork',
             hessian_structure='full',subnetwork_indices=subnetwork_indices)
la.fit(train_loader)
pred = predict(test_loader,la, laplace=True)