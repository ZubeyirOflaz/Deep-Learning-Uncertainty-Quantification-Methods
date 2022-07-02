import torch
from laplace import Laplace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributions as dists
from netcal.metrics import ECE

from torch.utils.data import Dataset, random_split
import pickle
# import helper
from config import dataset_paths, models, casting_args
import os
from laplace.utils import LargestMagnitudeSubnetMask, ModuleNameSubnetMask

laplace_target = 'arrhythmia'


def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device='cuda')
            y = y.to(device='cuda')

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

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

transformations = transforms.Compose([transforms.Resize(int((image_resolution + 1) / 2) * 3),
                                      transforms.RandomCrop(image_resolution),
                                      transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

train_set = datasets.ImageFolder(casting_train_path, transform=transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True
                                           , num_workers=num_workers, pin_memory=True)

test_set = datasets.ImageFolder(casting_test_path, transform=transformations)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

check_accuracy(test_loader, casting_model)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

probs_map = predict(test_loader, casting_model, laplace=False)
# _, prediction = probs_map.max(1)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
# ece_map = ECE(bins=2).measure(probs_map.detach().numpy(), targets.detach().numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
print(f'[MAP] Acc.: {acc_map:.1%}; NLL: {nll_map:.3}')

torch.cuda.empty_cache()
subnetwork_mask = LargestMagnitudeSubnetMask(casting_model, n_params_subnet=256)
subnetwork_indices = subnetwork_mask.select()

subnetwork_indices = torch.LongTensor(subnetwork_indices.cpu())
# subnetwork_mask = ModuleNameSubnetMask(casting_model, module_names=['layer.9', 'layer.12'])
# subnetwork_mask.select()
# subnetwork_indices = subnetwork_mask.indices

la = Laplace(casting_model, likelihood='classification', subset_of_weights='subnetwork',
             hessian_structure='full', subnetwork_indices=subnetwork_indices)
la.fit(train_loader)

pred = predict(test_loader, la, laplace=True)

probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
# ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(f'[Laplace] Acc.: {acc_laplace:.1%} NLL: {nll_laplace:.3}')


