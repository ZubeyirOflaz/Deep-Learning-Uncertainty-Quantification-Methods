import optuna
from optuna.trial import TrialState
import pandas as pd
from config import dataset_paths, models
import pickle
import numpy
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributions as dists
from sklearn import preprocessing
from utils.helper import weighted_classes_arrhythmia
import torcheck

# Loading and preprocessing datasets, inputting some of the hyperparameters

'''ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
arrhythmia_train_path = ROOT_DIR + dataset_paths['arrhythmia_train']
arrhythmia_test_path = ROOT_DIR + dataset_paths['arrhythmia_test']

arrhythmia_model_path = models['base_models']['arrhythmia']

with open(arrhythmia_model_path, "rb") as fin:
    arrhythmia_model = pickle.load(fin)

with open(arrhythmia_train_path, 'rb') as fin:
    arrhythmia_train = numpy.load(fin, allow_pickle=True)

with open(arrhythmia_test_path, 'rb') as fin:
    arrhythmia_test = numpy.load(fin, allow_pickle=True)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
ensemble_num = 4
batch_size = 16
num_workers = 0
num_epochs = 20

params = {'batch_size': batch_size,
          'num_workers': num_workers}

train_x = arrhythmia_train[:, 0]
train_y = arrhythmia_train[:, 1]

train_x = torch.tensor([i.numpy() for i in train_x]).to(torch.device('cuda'))
train_y = torch.tensor([i.numpy() for i in train_y]).to(torch.device('cuda')).flatten()
train = TensorDataset(train_x, train_y)
train_loader = [DataLoader(train, shuffle=True, **params) for _ in range(ensemble_num)]

hidden_dims = train_x.size()[1]
num_categories = int(train_y.max() + 1)

test_x = arrhythmia_test[:, 0]
test_y = arrhythmia_test[:, 1]

test_x = torch.tensor([i.numpy() for i in test_x]).to(torch.device('cuda'))
test_y = torch.tensor([i.numpy() for i in test_y]).to(torch.device('cuda')).flatten()
test = TensorDataset(test_x, test_y)

test_loader = DataLoader(test, shuffle=False, **params)'''


# Dataloader for pandas dataframe to pytorch dataset conversion
class pandas_dataset(Dataset):

    def __init__(self, pd_dataframe):
        df = pd_dataframe

        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1:].values
        transformer = preprocessing.RobustScaler().fit(x)
        x = transformer.transform(x)

        self.x_train = torch.tensor(x.astype(numpy.float32), dtype=torch.float32)
        self.y_train = torch.tensor(y.astype(numpy.int8), dtype=torch.long)
        y = torch.tensor(y[:, 0])

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


arrhythmia_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data'
arrhythmia_classes = 'http://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.names'
dataset = pd.read_csv(arrhythmia_data, header=None)

dataset = dataset.replace('?', numpy.nan)
dataset.fillna(dataset.median(), inplace=True)
dataset[279] = dataset[279] - 1

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
ensemble_num = 4
batch_size = 8
num_workers = 0
num_epochs = 50
hidden_dims = 279
num_categories = 16


params = {'batch_size': batch_size,
          'num_workers': num_workers}

df_dataset = pandas_dataset(dataset)
train, test = random_split(df_dataset, [400, 52], generator=torch.Generator().manual_seed(42))


train_sample_dist = weighted_classes_arrhythmia(train)
train_weights = torch.DoubleTensor(train_sample_dist)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

test_sample_dist = weighted_classes_arrhythmia(test)
test_weights = torch.DoubleTensor(test_sample_dist)
test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))

train_loader = [DataLoader(train, sampler=train_sampler, **params) for _ in range(ensemble_num)]
test_loader = DataLoader(test, sampler=test_sampler, **params)

arrhythmia_model_path = models['base_models']['arrhythmia']

'''with open(arrhythmia_model_path, "rb") as fin:
    arrhythmia_model = pickle.load(fin)


def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py)


targets = torch.cat([y for x, y in test_loader], dim=0).to(device)
targets = torch.reshape(targets, (-1,))

probs_map = predict(test_loader, arrhythmia_model.to(device))
acc_map = (probs_map.argmax(-1) == targets).float().mean()
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
print(f'[MAP] Acc.: {acc_map:.1%}; NLL: {nll_map:.3}')
print(arrhythmia_model)'''
'''transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("../data", train=False, transform=transform)
print('stage3')

train_loader = [
    DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    for _ in range(ensemble_num)
]
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
'''


def optuna_model(trial):
    # Main MIMO model layer
    class MIMOModel(nn.Module):
        def __init__(self, hidden_dim: int = hidden_dims, ensemble_num: int = 3):
            super(MIMOModel, self).__init__()
            self.output_dim = trial.suggest_int('output_dim', 32, 1024)
            # self.input_layer = nn.Linear(hidden_dim * ensemble_num, hidden_dim * ensemble_num)
            self.backbone_model = BackboneModel(hidden_dim, ensemble_num, self.output_dim)
            self.ensemble_num = ensemble_num
            self.output_layer = nn.Linear(self.output_dim, num_categories * ensemble_num)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            batch_size = list(input_tensor.size())[0]
            ensemble_num = self.ensemble_num
            # input_tensor = self.input_layer(input_tensor)

            # usual model forward
            output = self.backbone_model(input_tensor)  # (batch_size, output_dim)
            output = self.output_layer(output)  # (batch_size, 16 * ensemble_num)
            output = output.reshape(
                batch_size, ensemble_num, -1#, ensemble_num
            )  # (batch_size, ensemble_num, 16, ensemble_num)
            #output = torch.diagonal(output, offset=0, dim1=1, dim2=3).transpose(2, 1)  # (batch_size, ensemble_num, 16)
            output = F.log_softmax(output, dim=-1)  # (batch_size, ensemble_num, 16)
            return output

    # Middle layer for the MIMO model that is mainly configured by optuna
    class BackboneModel(nn.Module):

        def __init__(self, hidden_dim: int, ensemble_num: int, output_dim: int):
            super(BackboneModel, self).__init__()
            layers = []
            in_features = hidden_dim * ensemble_num
            num_layers = trial.suggest_int('num_layers', 1, 5)
            for i in range(num_layers):
                out_dim = trial.suggest_int('n_units_l{}'.format(i), 8, 1024)
                layers.append(nn.Linear(in_features, out_dim))
                layers.append(nn.ReLU())
                dropout_rate = trial.suggest_float('dr_rate_l{}'.format(i), 0.0, 0.5)
                layers.append(nn.Dropout(dropout_rate))
                in_features = out_dim
            layers.append(nn.Linear(in_features, output_dim))
            layers.append(nn.ReLU())
            self.layers = layers
            self.module = nn.Sequential(*self.layers).to(device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = self.module(x)
            return output

    mimo_optuna = MIMOModel(hidden_dim=hidden_dims, ensemble_num=ensemble_num)
    return mimo_optuna


bug_dict = {}


def objective(trial):
    # Model and main parameter initialization

    model = optuna_model(trial=trial).to(device)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
    optimizer = getattr(optim, 'Adadelta')(model.parameters(), lr=lr)
    gamma = trial.suggest_float('gamma', 0.7, 1.0)
    scheduler = StepLR(optimizer, step_size=len(train_loader[0]), gamma=gamma)
    '''torcheck.register(optimizer)
    torcheck.add_module_changing_check(model)
    torcheck.add_module_nan_check(model)
    torcheck.add_module_inf_check(model)'''
    # Training and eval loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for datum in zip(*train_loader):
            # Training
            # bug_dict['datum'] = datum
            model_inputs = torch.cat([data[0] for data in datum], dim=1).to(device)
            # model_inputs = model_inputs[:, :, None, :]
            # bug_dict['model_inputs'] = model_inputs
            targets = torch.stack([data[1] for data in datum]).to(device)
            # bug_dict['t_targets'] = targets
            #targets = targets.squeeze()
            # bug_dict['t_targets_a'] = targets
            batch_size = list(targets.size())[1]
            optimizer.zero_grad()
            outputs = model(model_inputs)
            # bug_dict['t_outputs'] = outputs
            '''print('Preds Before')
            print(outputs.size())'''
            # outputs = outputs.reshape(ensemble_num * batch_size, -1)
            # targets = targets.reshape(ensemble_num * batch_size)
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
        correct = 0
        with torch.no_grad():
            for data in test_loader:

                model_inputs = torch.cat([data[0]] * ensemble_num, dim=1).to(device)

                # model_inputs = model_inputs[:, :, None, :]
                target = data[1].to(device)
                target = target.squeeze()
                # bug_dict['data'] = data
                outputs = model(model_inputs)
                # bug_dict['outputs'] = outputs
                output = torch.mean(outputs, axis=1)
                # bug_dict['output'] = output
                # print(output)
                # print(target)
                # bug_dict['target'] = target
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=-1, keepdim=True)
                # target_test = target.view_as(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = 100.0 * correct / len(test_loader.dataset)
        print(f'epoch {epoch}: {acc}')
        trial.report(acc, epoch)
        '''if trial.should_prune():
            raise optuna.exceptions.TrialPruned()'''

    return acc


# study = optuna.create_study(direction="maximize")
study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=100, multivariate=True
                                                               , warn_independent_sampling=False), direction='maximize')
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
