import gc
import os

import pandas as pd

from config import dataset_paths as args
from config import models as model_paths
import torch
from utils.ford_a_dataloader import ARFFDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.ford_a_optuna_models import optuna_ford_a_mimo
import pickle
import random
from utils.helper import create_study_analysis, load_mimo_model, \
    MimoTrainValidateFordA, MimoTrainValidateCasting, weighted_classes
from utils.evaluation_metrics import calculate_metric_mimo, create_metric_dataframe, get_metrics, get_runtime_model_size
from utils.mimo_archive import mimo_cnn_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '\\'.join(ROOT_DIR.split('\\')[:-1])
train_set_path = ROOT_DIR + args['ford_a_train']
test_set_path = ROOT_DIR + args['ford_a_test']

batch_size = 8
num_workers = 0
ensemble_num = 3
num_categories = 2

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model_dict = {'ensemble_num': ensemble_num,
              'device': device}

train_dataset = ARFFDataset(train_set_path, data_scaling=False)
test_dataset = ARFFDataset(test_set_path, data_scaling=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers
                          , pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
train_dataset = None
test_dataset = None

study_number = 8966978
model_instance = 166
model_path = model_paths['mimo_models']['ford_a_model']
study_path = model_paths['mimo_models']['ford_a_study']
with open(f'{ROOT_DIR}\\{study_path}', 'rb') as fin:
    study = pickle.load(fin)
study_df = create_study_analysis(study.get_trials())

model = optuna_ford_a_mimo(study.get_trials(deepcopy=True)[model_instance], model_dict).to(device)
model.load_state_dict(torch.load(f'{ROOT_DIR}\\{model_path}'))
model_params = study_df.loc[model_instance]

preds, targets = get_runtime_model_size(test_loader,model,batch_size=batch_size,model_type='ford_a')
results_dict = calculate_metric_mimo(model, test_loader, 3, 'ford_a')
get_metrics(torch.from_numpy(results_dict['means']),torch.from_numpy(results_dict['targets']))
ford_a_test_dataframe = create_metric_dataframe(results_dict, True)
ford_a_test_dataframe['dataset'] = 'test'
results_dict = calculate_metric_mimo(model, train_loader, 3, 'ford_a')
get_metrics(torch.from_numpy(results_dict['means']),torch.from_numpy(results_dict['targets']))
ford_a_train_dataframe = create_metric_dataframe(results_dict, True)
ford_a_train_dataframe['dataset'] = 'train'
ford_a_dataframe = pd.concat([ford_a_train_dataframe, ford_a_test_dataframe], ignore_index=True)
grouped_dataframe = pd.pivot_table(ford_a_dataframe, index=['accuracy', 'dataset'],
                                   values=['m_0', 'm_1', 'std_0', 'std_1', 'total_divergence'],
                                   aggfunc=['mean', 'min']).reset_index()
# model = load_mimo_model(8966978, 169, model_dict= model_dict)
gc.collect()

# Mimo benchmarking for casting dataset
train_set_path = ROOT_DIR + args['casting_train']
test_set_path = ROOT_DIR + args['casting_test']

batch_size = 4
image_resolution = 63
num_workers = 0
ensemble_num = 3
num_categories = 2
study_name = str(random.randint(100000, 999999))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

transformations = transforms.Compose([transforms.Resize(int((image_resolution + 1) * 1.40)),
                                      transforms.RandomCrop(image_resolution),
                                      transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

train_set = datasets.ImageFolder(train_set_path, transform=transformations)
train_sample_dist = weighted_classes(train_set.imgs, len(train_set.classes))
train_weights = torch.DoubleTensor(train_sample_dist)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
train_loader = [torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers,
                                            pin_memory=True,sampler=train_sampler)
                for _ in range(ensemble_num)]

test_set = datasets.ImageFolder(test_set_path, transform=transformations)
test_sample_dist = weighted_classes(test_set.imgs, len(test_set.classes))
test_weights = torch.DoubleTensor(test_sample_dist)
test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=test_sampler)

train_loader_n = [torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers,
                                            pin_memory=True)
                for _ in range(ensemble_num)]

test_loader_n = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


model_instance = 135
model_path = model_paths['mimo_models']['casting_model']
study_path = model_paths['mimo_models']['casting_study']
with open(f'{ROOT_DIR}\\{study_path}', 'rb') as fin:
    casting_study = pickle.load(fin)
study_df = create_study_analysis(casting_study.get_trials())

model = mimo_cnn_model(casting_study.get_trials(deepcopy = True)[135])
model.load_state_dict(torch.load(f'{ROOT_DIR}\\{model_path}'))
class_hyperparameters = study_df.loc[casting_study.best_trial.number]
class_hyperparameters.lr = class_hyperparameters.lr * 0.8
#class_hyperparameters.gamma = 0.98
casting_mimo = MimoTrainValidateCasting(model, class_hyperparameters, train_loader, test_loader)
#casting_mimo.model_train(num_epochs = 90)
'''model_trained = casting_mimo.model




casting_mimo_trained = MimoTrainValidateCasting(model,class_hyperparameters,
                                                train_loader_n,test_loader_n)
casting_mimo_trained.model_validate()
model = casting_mimo_trained.model
'''
#casting_mimo.model_validate()
#torch.save(model.state_dict(),'casting_mimo.pyt')

preds, targets = get_runtime_model_size(test_loader_n,model,batch_size=batch_size,model_type='casting')

train_loader_n = torch.utils.data.DataLoader(train_set, batch_size=256,
                                            num_workers=num_workers,
                                            pin_memory=True)

results_dict = calculate_metric_mimo(model, test_loader_n, 3, 'casting')
get_metrics(torch.from_numpy(results_dict['means']),torch.from_numpy(results_dict['targets']))
casting_test_dataframe = create_metric_dataframe(results_dict, True)
casting_test_dataframe['dataset'] = 'test'
results_dict = calculate_metric_mimo(model, train_loader_n, 3, 'casting')
get_metrics(torch.from_numpy(results_dict['means']),torch.from_numpy(results_dict['targets']))
casting_train_dataframe = create_metric_dataframe(results_dict, True)
casting_train_dataframe['dataset'] = 'train'
casting_dataframe = pd.concat([casting_train_dataframe, casting_test_dataframe], ignore_index=True)
grouped_dataframe = pd.pivot_table(casting_dataframe, index=['accuracy', 'dataset'],
                                   values=['m_0', 'm_1', 'std_0', 'std_1', 'total_divergence'],
                                   aggfunc=['mean', 'min']).reset_index()