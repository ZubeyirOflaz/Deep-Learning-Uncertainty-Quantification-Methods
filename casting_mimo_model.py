import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
# import helper
import logging
import time
import os

import optuna
from optuna.trial import TrialState

from config import dataset_paths as args
import random

# Data Import

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['casting_train']
test_set_path = ROOT_DIR + args['casting_test']

batch_size = 128
image_resolution = 127
num_workers = 0
ensemble_num = 4
num_categories = 2

LOG_INTERVAL = 10
EPOCHS = 25

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

random_seed = 15

torch.random.manual_seed(random_seed)
random.seed(random_seed)
transformations = transforms.Compose([transforms.Resize(int((image_resolution + 1) * 1.40)),
                                      transforms.RandomCrop(image_resolution),
                                      transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

train_set = datasets.ImageFolder(train_set_path, transform=transformations)
train_loader = [torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers, pin_memory=True)
                for _ in range(ensemble_num)]

torch.random.manual_seed(random_seed)
random.seed(random_seed)

test_set = datasets.ImageFolder(test_set_path, transform=transformations)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

N_TRAIN_EXAMPLES = len(train_set)
N_TEST_EXAMPLES = len(test_set)


def mimo_cnn_model(trial):

    class MimoCnnModel(nn.Module):
        def __init__(self, ensemble_num: int, num_categories: int):
            super(mimo_cnn_model, self).__init__()
            self.output_dim = trial.suggest_int('output_dim', 32, 256)
            self.num_channels = trial.suggest_int('num_channels', 4, 24) * ensemble_num
            self.final_img_resolution = 12 * trial.suggest_int('img_multiplier', 1, 3)
            self.input_dim = self.num_channels * (self.final_img_resolution ^ 2)
            self.conv_module = ConvModule(self.num_channels, self.final_img_resolution, ensemble_num)
            self.linear_module = LinearModule(self.input_dim, self.output_dim)
            self.output_layer = nn.Linear(self.output_dim, num_categories * ensemble_num)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            size_list = list(input_tensor.size())
            ensemble_num, batch_size, *_ = size_list
            conv_result = self.conv_module(input_tensor.reshape(size_list[1:-1] + [size_list[-1] * size_list[0]]))
            output = self.linear_module(conv_result.reshape(batch_size, ensemble_num, -1))
            output = self.output_layer(output)
            output = output.reshape(
                batch_size, ensemble_num, -1, ensemble_num
            )  # (batch_size, ensemble_num, num_categories, ensemble_num)
            output = torch.diagonal(output, offset=0, dim1=1, dim2=3).transpose(2,
                                                                                1)  # (batch_size, ensemble_num, num_categories)
            output = F.log_softmax(output, dim=-1)  # (batch_size, ensemble_num, num_categories)
            return output

    class ConvModule(nn.Module):
        def __init__(self, num_channels: int, final_img_resolution: int, ensemble_num: int):
            super(ConvModule, self).__init__()
            layers = []
            num_layers = trial.suggest_int('num_cnn_layers', 1, 2)
            input_channels = 1
            for i in num_layers:
                num_filters = trial.suggest_categorical(f'num_filters_{i}', [16, 32, 48, 64])
                kernel_size = trial.suggest_int(f'kernel_size_{i}', 2, 4)
                layers.append(nn.Conv2d(input_channels, num_filters, kernel_size))
                layers.append(nn.MaxPool2d(4, 2))
                input_channels = num_filters
            layers.append(nn.Conv2d(input_channels, num_channels, 3))
            layers.append(nn.AdaptiveMaxPool2d((final_img_resolution, final_img_resolution * ensemble_num)))
            self.layers = layers

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            module = nn.Sequential(*self.layers).to(device)
            output = module(input_tensor)
            return output

    class LinearModule(nn.Module):
        def __init__(self, input_dimension: int, output_dimension: int):
            super(LinearModule, self).__init__()
            layers = []
            in_features = input_dimension
            num_layers = trial.suggest_int('num_layers', 1, 3)
            for i in range(num_layers):
                out_dim = trial.suggest_int('n_units_l{}'.format(i), 8, 1024)
                layers.append(nn.Linear(in_features, out_dim))
                layers.append(nn.ReLU())
                dropout_rate = trial.suggest_float('dr_rate_l{}'.format(i), 0.0, 0.5)
                if dropout_rate > 0.05:
                    layers.append(nn.Dropout(dropout_rate))
                in_features = out_dim
            layers.append(nn.Linear(in_features, output_dimension))
            layers.append(nn.ReLU())
            self.layers = layers

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            module = nn.Sequential(*self.layers).to(device)
            output = module(x)
            return output
    mimo_optuna = MimoCnnModel(ensemble_num=ensemble_num,num_categories=num_categories)

    return mimo_optuna

