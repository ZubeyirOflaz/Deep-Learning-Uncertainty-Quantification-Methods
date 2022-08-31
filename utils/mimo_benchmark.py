import gc
import os

import pandas as pd

from config import dataset_paths as args
import torch
from utils.ford_a_dataloader import ARFFDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.ford_a_optuna_models import optuna_ford_a_mimo
import pickle
import random
from utils.helper import create_study_analysis, load_mimo_model, \
    MimoTrainValidateFordA, MimoTrainValidateCasting, weighted_classes
from utils.evaluation_metrics import calculate_metric_mimo, create_metric_dataframe
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

with open(f'{ROOT_DIR}\\model_repo\\study_{study_number}.pkl', 'rb') as fin:
    study = pickle.load(fin)
study_df = create_study_analysis(study.get_trials())

model = optuna_ford_a_mimo(study.get_trials(deepcopy=True)[model_instance], model_dict).to(device)
model.load_state_dict(torch.load(f'{ROOT_DIR}\\model_repo\\{model_instance}_{study_number}.pyt'))
model_params = study_df.loc[model_instance]

#model_class = MimoTrainValidateFordA(model, model_params, trainloader=train_loader, testloader=test_loader)
#preds, targets = model_class.model_validate(get_predictions=True)

results_dict = calculate_metric_mimo(model,test_loader,3,'ford_a')
ford_a_test_dataframe = create_metric_dataframe(results_dict, True)
ford_a_test_dataframe['dataset'] = 'test'
results_dict = calculate_metric_mimo(model,train_loader,3,'ford_a')
ford_a_train_dataframe = create_metric_dataframe(results_dict,True)
ford_a_train_dataframe['dataset'] = 'train'
ford_a_dataframe = pd.concat([ford_a_train_dataframe,ford_a_test_dataframe], ignore_index= True)
grouped_dataframe = pd.pivot_table(ford_a_dataframe, index = ['accuracy', 'dataset'],
                                   values =['m_0','m_1','std_0','std_1','total_divergence'],
                                   aggfunc = ['mean','min']).reset_index()
# model = load_mimo_model(8966978, 169, model_dict= model_dict)
gc.collect()

#train_set_path = ROOT_DIR + args['casting_train']
test_set_path = ROOT_DIR + args['casting_test']

batch_size = 8
image_resolution = 63
num_workers = 0
ensemble_num = 3
num_categories = 2
study_name = str(random.randint(100000, 999999))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

transformations = transforms.Compose([transforms.Resize(int((image_resolution + 1) * 1.10)),
                                      transforms.RandomCrop(image_resolution),
                                      transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

train_set = datasets.ImageFolder(train_set_path, transform=transformations)
train_sample_dist = weighted_classes(train_set.imgs, len(train_set.classes))
train_weights = torch.DoubleTensor(train_sample_dist)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
train_loader = [torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers, sampler=train_sampler,
                                            pin_memory=True)
                for _ in range(ensemble_num)]

test_set = datasets.ImageFolder(test_set_path, transform=transformations)
test_sample_dist = weighted_classes(test_set.imgs, len(test_set.classes))
test_weights = torch.DoubleTensor(test_sample_dist)
test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=num_workers)




casting_study_path = 'N:\Thesis\Models\casting_mimo_study.pkl'
casting_model_path = 'N:\Thesis\Models\casting_mimo_model.pyt'

with open(casting_study_path, 'rb') as fin:
    casting_study = pickle.load(fin)
study_df = create_study_analysis(study.get_trials())

model = mimo_cnn_model(casting_study.best_trial)
model.load_state_dict(torch.load(casting_model_path))
