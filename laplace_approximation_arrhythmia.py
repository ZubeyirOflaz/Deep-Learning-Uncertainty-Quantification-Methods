import numpy as np
import torch
from laplace import Laplace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributions as dists
from netcal.metrics import ECE
import pandas
import numpy

from torch.utils.data import Dataset, random_split
import pickle
# import helper
from config import dataset_paths, models, casting_args
import os
from laplace.utils import LargestMagnitudeSubnetMask, ModuleNameSubnetMask

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

arrhythmia_train = arrhythmia_train.tolist()
train_loader = DataLoader(torch.tensor(arrhythmia_train),shuffle=True, **params)
test_loader = DataLoader(arrhythmia_test,shuffle=False,**params)

check_accuracy(train_loader,arrhythmia_model)