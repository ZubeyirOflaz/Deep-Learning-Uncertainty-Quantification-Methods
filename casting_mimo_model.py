import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
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

batch_size = 8
image_resolution = 127
num_workers = 0
ensemble_num = 3
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
            super(MimoCnnModel, self).__init__()
            self.output_dim = trial.suggest_int('output_dim', 32, 256)
            self.num_channels = trial.suggest_int('num_channels', 4, 24) * ensemble_num
            self.final_img_resolution = 12 # * trial.suggest_int('img_multiplier', 1, 3)
            self.input_dim = self.num_channels * (self.final_img_resolution * self.final_img_resolution)
            self.conv_module = ConvModule(self.num_channels, self.final_img_resolution, ensemble_num)
            self.linear_module = LinearModule(self.input_dim, self.output_dim)
            self.output_layer = nn.Linear(self.output_dim, num_categories * ensemble_num)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            size_list = list(input_tensor.size())
            ensemble_num, batch_size, *_ = size_list
            conv_result = self.conv_module(input_tensor.reshape(size_list[1:-1] + [size_list[-1] * size_list[0]]))
            '''print(self.input_dim)
            print(conv_result.size())
            print(conv_result.reshape(batch_size, ensemble_num, -1).size())'''
            output = self.linear_module(conv_result.flatten()) #.reshape(batch_size, ensemble_num, -1))
            print('tensor shapes')
            print(output.size())
            output = self.output_layer(output)
            print(output.size())
            output = output.reshape(
                batch_size, ensemble_num, -1, ensemble_num
            )  # (batch_size, ensemble_num, num_categories, ensemble_num)
            print(output.size())
            output = torch.diagonal(output, offset=0, dim1=1, dim2=3).transpose(2,
                                                                                1)  # (batch_size, ensemble_num, num_categories)
            print(output.size())
            output = F.log_softmax(output, dim=-1)  # (batch_size, ensemble_num, num_categories)
            print(output.size())
            return output

    class ConvModule(nn.Module):
        def __init__(self, num_channels: int, final_img_resolution: int, ensemble_num: int):
            super(ConvModule, self).__init__()
            layers = []
            num_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            input_channels = 1
            for i in range(num_layers):
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
                out_dim = trial.suggest_int('n_units_l{}'.format(i), 8, 512)
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

def objective(trial):
    # Model and main parameter initialization

    model = mimo_cnn_model(trial=trial).to(device)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
    optimizer = getattr(optim, 'Adadelta')(model.parameters(), lr=lr)
    gamma = trial.suggest_float('gamma', 0.0, 1.0)
    scheduler = StepLR(optimizer, step_size=len(train_loader[0]), gamma=gamma)
    num_epochs = 20

    # Training and eval loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for datum in zip(*train_loader):
            # Training
            #bug_dict['datum'] = datum
            model_inputs = torch.stack([data[0] for data in datum]).to(device)
            #model_inputs = model_inputs[:, :, None, :]
            #bug_dict['model_inputs'] = model_inputs
            targets = torch.stack([data[1] for data in datum]).to(device)
            #bug_dict['t_targets'] = targets
            #bug_dict['t_targets_a'] = targets
            optimizer.zero_grad()
            outputs = model(model_inputs)
            ensemble_num, batch_size = list(targets.size())
            #bug_dict['t_outputs'] = outputs

            loss = F.nll_loss(
                outputs.reshape(ensemble_num * batch_size, -1), targets.reshape(ensemble_num * batch_size)
            )
            train_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'{epoch}: {loss}')
        '''print('Target Tensor')
        print(targets.reshape(ensemble_num * batch_size))
        print(targets.reshape(ensemble_num * batch_size).size())
        print('Pred Tensor')
        print(outputs.reshape(ensemble_num * batch_size, -1))
        print(outputs.reshape(ensemble_num * batch_size, -1).size())'''
        # Evaluation
        # print(f'train loss: {train_loss/}')
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                model_inputs = torch.stack([data[0]] * ensemble_num).to(device)
                #model_inputs = model_inputs[:, :, None, :]
                target = data[1].to(device)
                #bug_dict['data'] = data
                outputs = model(model_inputs)
                #bug_dict['outputs'] = outputs
                output = torch.mean(outputs, axis=1)
                #bug_dict['output'] = output
                # print(output)
                # print(target)
                #bug_dict['target'] = target
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=-1, keepdim=True)
                #target_test = target.view_as(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = 100.0 * correct / len(test_loader.dataset)
        print(f'epoch {epoch}: {acc}')
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    '''with open("model_repo\\{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)'''

    return acc

study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True,
                                                               group=True, constant_liar= True),
                            direction='maximize')
study.optimize(objective, n_trials=200)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

'''with open("model_repo\\{}.pickle".format(study.best_trial.number), "rb") as fin:
    best_result = pickle.load(fin)
with open(f'model_repo\\best_models\\{study.best_trial.value}.pickle', 'wb') as fout:
    pickle.dump(best_result, fout)'''