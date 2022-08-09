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

# Data Import

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['casting_train']
test_set_path = ROOT_DIR + args['casting_test']

batch_size = 8
image_resolution = 63
num_workers = 0
ensemble_num = 3
num_categories = 2
study_name = str(random.randint(100000,999999))
LOG_INTERVAL = 10

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

N_TRAIN_EXAMPLES = len(train_set)
N_TEST_EXAMPLES = len(test_set)

model2 = None


def mimo_cnn_model(trial):
    global model2

    class MimoCnnModel(nn.Module):
        def __init__(self, ensemble_num: int, num_categories: int):
            super(MimoCnnModel, self).__init__()
            self.output_dim = trial.suggest_int('output_dim', 32, 1024)
            self.num_channels = trial.suggest_int('num_channels', 8, 32) * ensemble_num
            self.final_img_resolution = 8
            self.input_dim = self.num_channels * (self.final_img_resolution * self.final_img_resolution) * ensemble_num
            self.conv_module = ConvModule(self.num_channels, self.final_img_resolution, ensemble_num)
            self.linear_module = LinearModule(self.input_dim, self.output_dim)
            self.output_layer = nn.Linear(self.output_dim, num_categories * ensemble_num * ensemble_num)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            batch_size = input_tensor.size()[0]
            conv_result = self.conv_module(input_tensor)
            #print(self.input_dim)
            # print(conv_result.size())
            #print(conv_result.reshape(batch_size, -1).size())
            output = self.linear_module(conv_result.reshape(batch_size, -1))
            # print('tensor shapes')
            # print(output.size())
            output = self.output_layer(output)
            # print(output.size())
            output = output.reshape(
                batch_size, ensemble_num, -1, ensemble_num
            )  # (batch_size, ensemble_num, num_categories, ensemble_num)
            # print(output.size())
            output = torch.diagonal(output, offset=0, dim1=1, dim2=3).transpose(2, 1)
            # print(output.size())
            output = F.log_softmax(output, dim=-1)  # (batch_size, ensemble_num, num_categories)
            # print(output.size())
            return output

    class ConvModule(nn.Module):
        def __init__(self, num_channels: int, final_img_resolution: int, ensemble_num: int):
            super(ConvModule, self).__init__()
            layers = []
            num_layers = trial.suggest_int('num_cnn_layers', 4, 4)
            input_channels = 1
            for i in range(num_layers):
                num_filters = trial.suggest_categorical(f'num_filters_{i}', [4, 8, 16])
                kernel_size = trial.suggest_int(f'kernel_size_{i}', 2, 4)
                layers.append(nn.Conv2d(input_channels, num_filters, ((kernel_size * (i+1)), kernel_size * ensemble_num)))
                if i < 1:
                    pool_stride = 2
                else:
                    pool_stride = 1
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d((2, 2 * ensemble_num), pool_stride))
                input_channels = num_filters
            layers.append(nn.Conv2d(input_channels, input_channels * 2, (3, 3 * ensemble_num)))
            layers.append(nn.ReLU())
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
            num_layers = 1  # trial.suggest_int('num_layers', 1, 3)
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

    mimo_optuna = MimoCnnModel(ensemble_num=ensemble_num, num_categories=num_categories)
    model2 = mimo_optuna
    return mimo_optuna


def objective(trial):
    # Model and main parameter initialization

    model = mimo_cnn_model(trial=trial).to(device)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
    gamma = trial.suggest_float('gamma', 0.5, 1)
    scheduler = StepLR(optimizer, step_size=(len(train_loader[0])), gamma=gamma)

    num_epochs = 20
    '''torcheck.register(optimizer)
    torcheck.add_module_changing_check(model)
    torcheck.add_module_nan_check(model)
    torcheck.add_module_inf_check(model)
    torcheck.disable()'''
    # Training and eval loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for datum in zip(*train_loader):
            # Training
            # bug_dict['datum'] = datum
            model_inputs = torch.cat([data[0] for data in datum], dim=3).to(device)
            # model_inputs = model_inputs[:, :, None, :]
            # bug_dict['model_inputs'] = model_inputs
            targets = torch.stack([data[1] for data in datum]).to(device)
            # bug_dict['t_targets'] = targets
            # bug_dict['t_targets_a'] = targets
            optimizer.zero_grad()
            outputs = model(model_inputs)
            ensemble_num, batch_size = list(targets.size())
            # bug_dict['t_outputs'] = outputs

            loss = F.nll_loss(
                outputs.reshape(ensemble_num * batch_size, -1), targets.reshape(ensemble_num * batch_size)
            )
            train_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'{epoch}: {train_loss}')
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
        test_size = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                model_inputs = torch.cat([data[0]] * ensemble_num, dim=3).to(device)
                # model_inputs = model_inputs[:, :, None, :]
                target = data[1].to(device)
                # bug_dict['data'] = data
                outputs = model(model_inputs)
                # bug_dict['outputs'] = outputs
                output = torch.mean(outputs, axis=1)
                # bug_dict['output'] = output
                # print(output)
                # print(target)
                # bug_dict['target'] = target
                test_size += len(target)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=-1, keepdim=True)
                # target_test = target.view_as(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= test_size
        acc = 100.0 * correct / test_size
        print(f'epoch {epoch}: {acc}')
        trial.report(acc, epoch)
        if trial.should_prune(): # or (epoch > 4 and acc < 52):
            raise optuna.exceptions.TrialPruned()
    torch.save(model.state_dict(), f"model_repo\\{trial.number}_{study_name}.pyt")

    '''with open("model_repo\\{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)'''

    return acc


study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
                            direction='maximize', study_name=study_name)


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

best_model = mimo_cnn_model(study.get_trials(deepcopy=True)[study.best_trial.number])
best_model.load_state_dict(torch.load(f"model_repo\\{study.best_trial.number}_{study_name}.pyt"))
torch.save(best_model.state_dict(), f'model_repo\\best_models\\casting_{study.best_trial.value}_{study_name}.pyt')

trial_dataframe = create_study_analysis(study.get_trials(deepcopy=True))
with open(f'model_repo\\study_{study.study_name}.pkl', 'wb') as fout:
    pickle.dump(study, fout)