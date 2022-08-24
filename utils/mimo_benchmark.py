import gc
import os
from config import dataset_paths as args
import torch
import torch.nn.functional as F
from utils.ford_a_dataloader import ARFFDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time
import torcheck
import logging
import optuna
from utils.ford_a_optuna_models import optuna_ford_a_mimo
import pickle
import random
from optuna.trial import TrialState
from utils.helper import create_study_analysis, load_mimo_model, MimoTrainValidateFordA, MimoTrainValidateCasting
from utils.evaluation_metrics import calculate_metric_mimo

study_name = str(random.randint(8000000, 8999999))

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
train_loader = [DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers
                           , pin_memory=True) for _ in range(ensemble_num)]
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

# model = load_mimo_model(8966978, 169, model_dict= model_dict)

gc.collect()
