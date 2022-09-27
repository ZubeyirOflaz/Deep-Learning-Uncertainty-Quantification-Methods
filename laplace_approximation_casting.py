import torch
from laplace import Laplace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributions as dists
from netcal.metrics import ECE

from torch.utils.data import Dataset, random_split
import pickle
from utils.evaluation_metrics import predict
from config import dataset_paths, models, casting_args
import os
from laplace.utils import LargestMagnitudeSubnetMask, ModuleNameSubnetMask
import random
import torch.distributions as dists
from netcal.metrics import ECE
from utils.evaluation_metrics import predict, calculate_metric_laplace,\
    create_metric_dataframe, get_metrics, benchmark_laplace, get_runtime_model_size
import pandas as pd




    # Laplace approximation for casting dataset


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

casting_train_path = ROOT_DIR + dataset_paths['casting_train']
casting_test_path = ROOT_DIR + dataset_paths['casting_test']

casting_model_path = models['base_models']['casting']

with open(casting_model_path, "rb") as fin:
    casting_model = pickle.load(fin)

batch_size = 16
image_resolution = 127
num_workers = 0
random_seed = 15

torch.random.manual_seed(random_seed)
random.seed(random_seed)
transformations = transforms.Compose([transforms.Resize(int((image_resolution+1) * 1.40)),
                                      transforms.RandomCrop(image_resolution),
                                      transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

torch.random.manual_seed(random_seed)
random.seed(random_seed)
complete_set = datasets.ImageFolder(casting_train_path, transform=transformations)
all_index = list(range(0, len(complete_set), 1))
val_index = list(range(0, len(complete_set), 10))
train_index = [x for x in all_index if x not in val_index]

train_set = torch.utils.data.Subset(complete_set, train_index)
val_set = torch.utils.data.Subset(complete_set, val_index)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True
                          , num_workers=num_workers, pin_memory=True)

val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

torch.random.manual_seed(random_seed)
random.seed(random_seed)
test_set = datasets.ImageFolder(casting_test_path, transform=transformations)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

probs_map = predict(test_loader, casting_model, laplace=False)
# _, prediction = probs_map.max(1)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
# ece_map = ECE(bins=2).measure(probs_map.detach().numpy(), targets.detach().numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
print(f'[MAP] Acc.: {acc_map:.1%}; NLL: {nll_map:.3}')
get_metrics(probs_map,targets)
preds, targets = get_runtime_model_size(train_loader,casting_model,batch_size=batch_size)
torch.cuda.empty_cache()
# subnetwork_mask = LargestMagnitudeSubnetMask(casting_model, n_params_subnet=256)
# subnetwork_indices = subnetwork_mask.select()

subnetwork_mask = ModuleNameSubnetMask(casting_model, module_names=['15'])
subnetwork_mask.select()
subnetwork_indices = subnetwork_mask.indices

subnetwork_indices = torch.LongTensor(subnetwork_indices.cpu())

la = Laplace(casting_model, likelihood='classification', subset_of_weights='subnetwork',
             hessian_structure='full', subnetwork_indices=subnetwork_indices)
la.fit(train_loader)

probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
# ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()
print(f'[Laplace] Acc.: {acc_laplace:.1%} NLL: {nll_laplace:.3}')

la.optimize_prior_precision(method='CV', val_loader=val_loader, pred_type='nn')

probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
# ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()
get_metrics(probs_laplace,targets)

print(f'[Laplace] Acc.: {acc_laplace:.1%} NLL: {nll_laplace:.3}')

results_dict = calculate_metric_laplace(la,test_loader,100)
ford_a_test_dataframe = create_metric_dataframe(results_dict,mimo_metric=False)
ford_a_test_dataframe['dataset'] = 'test'
results_dict = calculate_metric_laplace(la,train_loader,100)
ford_a_train_dataframe = create_metric_dataframe(results_dict,mimo_metric=False)
ford_a_train_dataframe['dataset'] = 'train'
ford_a_dataframe = pd.concat([ford_a_train_dataframe,ford_a_test_dataframe], ignore_index= True)
ford_a_dataframe.to_csv('casting_mimo.csv')
