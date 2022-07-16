import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.trial import TrialState
from config import dataset_paths, models
import os
import pickle
import numpy
from torch.utils.data import DataLoader, TensorDataset


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

ensemble_num = 3
batch_size = 16
num_workers = 0
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

params = {'batch_size': batch_size,
          'num_workers': num_workers}

train_x = arrhythmia_train[:, 0]
train_y = arrhythmia_train[:, 1]

train_x = torch.tensor([i.numpy() for i in train_x]).to(torch.device('cuda'))
train_y = torch.tensor([i.numpy() for i in train_y]).to(torch.device('cuda')).flatten()
train = TensorDataset(train_x, train_y)
train_loader = [DataLoader(train, shuffle=True, **params) for _ in range(ensemble_num)]

test_x = arrhythmia_test[:,0]
test_y = arrhythmia_test[:,1]

test_x = torch.tensor([i.numpy() for i in test_x]).to(torch.device('cuda'))
test_y = torch.tensor([i.numpy() for i in test_y]).to(torch.device('cuda')).flatten()
test = TensorDataset(test_x, test_y)

test_loader = DataLoader(test, shuffle=False, **params)

class MIMOModel(nn.Module):
    def __init__(self, hidden_dim: int = 784, ensemble_num: int = 3):
        super(MIMOModel, self).__init__()
        self.input_layer = nn.Linear(hidden_dim, hidden_dim * ensemble_num)
        self.backbone_model = BackboneModel(hidden_dim, ensemble_num)
        self.ensemble_num = ensemble_num
        self.output_layer = nn.Linear(128, 10 * ensemble_num)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        ensemble_num, batch_size, *_ = list(input_tensor.size())
        input_tensor = input_tensor.transpose(1, 0).view(
            batch_size, ensemble_num, -1
        )  # (batch_size, ensemble_num, hidden_dim)
        input_tensor = self.input_layer(input_tensor)  # (batch_size, ensemble_num, hidden_dim * ensemble_num)

        # usual model forward
        output = self.backbone_model(input_tensor)  # (batch_size, ensemble_num, 128)
        output = self.output_layer(output)  # (batch_size, ensemble_num, 10 * ensemble_num)
        output = output.reshape(
            batch_size, ensemble_num, -1, ensemble_num
        )  # (batch_size, ensemble_num, 10, ensemble_num)
        output = torch.diagonal(output, offset=0, dim1=1, dim2=3).transpose(2, 1)  # (batch_size, ensemble_num, 10)
        output = F.log_softmax(output, dim=-1)  # (batch_size, ensemble_num, 10)
        return output


class BackboneModel(nn.Module):
    def __init__(self, hidden_dim: int, ensemble_num: int):
        super(BackboneModel, self).__init__()
        self.l1 = nn.Linear(hidden_dim * ensemble_num, 256)
        self.l2 = nn.Linear(256, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1)
        x = self.l2(x)
        x = F.relu(x)
        return x