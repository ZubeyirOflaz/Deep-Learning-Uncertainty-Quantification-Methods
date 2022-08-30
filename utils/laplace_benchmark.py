import gc
import os

import pandas as pd
import numpy as np

from config import dataset_paths as args
from config import models
import torch
from utils.ford_a_dataloader import ARFFDataset
from torch.utils.data import DataLoader, TensorDataset
from utils.ford_a_optuna_models import optuna_ford_a_mimo
import pickle
import random
from utils.helper import create_study_analysis, load_mimo_model, MimoTrainValidateFordA, MimoTrainValidateCasting
from utils.evaluation_metrics import calculate_metric_laplace, create_metric_dataframe

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '\\'.join(ROOT_DIR.split('\\')[:-1])


arrhythmia_train_path = ROOT_DIR + args['arrhythmia_train']
arrhythmia_test_path = ROOT_DIR + args['arrhythmia_test']

arrhythmia_model_path = ROOT_DIR + models['laplace_approximation']['arrhythmia']

with open(arrhythmia_model_path, "rb") as fin:
    model = pickle.load(fin)

with open(arrhythmia_train_path, 'rb') as fin:
    arrhythmia_train = np.load(fin, allow_pickle=True)

with open(arrhythmia_test_path, 'rb') as fin:
    arrhythmia_test = np.load(fin, allow_pickle=True)

#Loading and preprocessing datasets, inumpyuting some of the hyperparameters
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

results_dict = calculate_metric_laplace(model,test_loader,100)
arrhythmia_test_dataframe = create_metric_dataframe(results_dict,mimo_metric=False)
arrhythmia_test_dataframe['dataset'] = 'test'
results_dict = calculate_metric_laplace(model,train_loader,100)
arrhythmia_train_dataframe = create_metric_dataframe(results_dict,mimo_metric=False)
arrhythmia_train_dataframe['dataset'] = 'train'
arrhythmia_dataframe = pd.concat([arrhythmia_train_dataframe,arrhythmia_test_dataframe], ignore_index= True)
