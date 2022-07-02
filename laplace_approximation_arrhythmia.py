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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
arrhythmia_train_path = ROOT_DIR + dataset_paths['arrhythmia_train']
arrhythmia_test_path = ROOT_DIR + dataset_paths['arrhythmia_test']

arrhythmia_model_path = models['base_models']['arrhythmia']

with open(arrhythmia_model_path, "rb") as fin:
    arrhythmia_model = pickle.load(fin)

with open(arrhythmia_train_path, 'rb') as fin:
    arrhythmia_train = pickle.load(fin)

with open(arrhythmia_test_path, 'rb') as fin:
    arrhythmia_test = pickle.load(fin)

batch_size = 16
num_workers = 0
