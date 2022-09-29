
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import pickle
from utils.helper import weighted_classes, create_study_analysis
import os
import torcheck
import optuna
from optuna.trial import TrialState

from config import dataset_paths as args
import random

'''ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['casting_train']
test_set_path = ROOT_DIR + args['casting_test']
'''
batch_size = 4
image_resolution = 63
num_workers = 0
ensemble_num = 3
num_categories = 2
study_name = str(random.randint(100000, 999999))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

def mimo_cnn_model(trial):
    class MimoCnnModel(nn.Module):
        def __init__(self, ensemble_num: int, num_categories: int):
            super(MimoCnnModel, self).__init__()
            self.output_dim = trial.suggest_int('output_dim', 64, 1024)
            self.num_channels = trial.suggest_int('num_channels', 64, 256)
            self.final_img_resolution = 6
            self.input_dim = self.num_channels * ((self.final_img_resolution -2) *
                                                  ((self.final_img_resolution * ensemble_num)-8))
            self.conv_module = ConvModule(self.num_channels, self.final_img_resolution, ensemble_num)
            self.linear_module = LinearModule(self.input_dim, self.output_dim)
            self.output_layer = nn.Linear(self.output_dim, num_categories * ensemble_num)
            self.softmax = nn.LogSoftmax(dim=-1)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            batch_size = input_tensor.size()[0]
            conv_result = self.conv_module(input_tensor)
            # print(self.input_dim)
            # print(conv_result.size())
            # print(conv_result.reshape(batch_size, -1).size())
            output = self.linear_module(conv_result.reshape(batch_size, -1))
            # print('tensor shapes')
            # print(output.size())
            output = self.output_layer(output)
            # print(output.size())
            output = output.reshape(
                batch_size, ensemble_num, -1
            )  # (batch_size, ensemble_num, num_categories)
            # print(output.size())
            # output = torch.diagonal(output, offset=0, dim1=1, dim2=3).transpose(2, 1)
            # print(output.size())
            output = self.softmax(output)  # (batch_size, ensemble_num, num_categories)
            # print(output.size())
            return output

    class ConvModule(nn.Module):
        def __init__(self, num_channels: int, final_img_resolution: int, ensemble_num: int):
            super(ConvModule, self).__init__()
            layers = []
            num_layers = trial.suggest_int('num_cnn_layers', 2,2)
            cnn_dropout = trial.suggest_discrete_uniform('drop_out_cnn', 0.05, 0.5, 0.05)
            input_channels = 1
            for i in range(num_layers):
                filter_base = [4, 8, 16, 32,64]
                filter_selections = [y * (i + 1) for y in filter_base]
                num_filters = trial.suggest_categorical(f'num_filters_{i}', filter_selections)
                kernel_size = trial.suggest_int(f'kernel_size_{i}', 3, 5)

                if i < 1:
                    pool_stride = 2
                else:
                    pool_stride = 1
                layers.append(nn.Conv2d(input_channels, num_filters, (kernel_size,
                                                                      (kernel_size * ensemble_num)),stride=pool_stride))
                if i < 1:
                    pool_stride = 2
                else:
                    pool_stride = 1
                if i < num_layers-1:
                    layers.append(nn.ReLU())
                    layers.append(nn.MaxPool2d((2, 2 * ensemble_num), pool_stride))
                    layers.append(nn.Dropout(cnn_dropout))
                input_channels = num_filters
            layers.append(nn.AdaptiveMaxPool2d((final_img_resolution, final_img_resolution * ensemble_num)))
            layers.append(nn.Conv2d(input_channels, num_channels, (3, 3 * ensemble_num)))
            self.layers = layers
            self.module = nn.Sequential(*self.layers).to(device)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            output = self.module(input_tensor)
            return output

    class LinearModule(nn.Module):
        def __init__(self, input_dimension: int, output_dimension: int):
            super(LinearModule, self).__init__()
            layers = []
            in_features = input_dimension
            num_layers = 1  # trial.suggest_int('num_layers', 1, 3)
            for i in range(num_layers):
                min_lim = int(512 / int(i+1))
                max_lim = int(2048 / (int(i+1)))
                out_dim = trial.suggest_int('n_units_l{}'.format(i), min_lim, max_lim)
                layers.append(nn.Linear(in_features, out_dim))
                layers.append(nn.ReLU())
                dropout_rate = trial.suggest_float('dr_rate_l{}'.format(i), 0.0, 0.5)
                if dropout_rate > 0.05:
                    layers.append(nn.Dropout(dropout_rate))
                in_features = out_dim
            layers.append(nn.Linear(in_features, output_dimension))
            layers.append(nn.ReLU())
            self.layers = layers
            self.module = nn.Sequential(*self.layers).to(device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = self.module(x)
            return output

    mimo_optuna = MimoCnnModel(ensemble_num=ensemble_num, num_categories=num_categories)
    return mimo_optuna
