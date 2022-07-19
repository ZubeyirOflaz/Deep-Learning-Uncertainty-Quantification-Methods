import configparser

import optuna
from optuna.trial import TrialState
import pandas as pd
from config import dataset_paths, models
import os
import pickle
import numpy
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from typing import NamedTuple
import torch.distributions as dists
from sklearn import preprocessing

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
dataset = pd.read_csv(arrhythmia_data,header = None)

dataset = dataset.replace('?', numpy.nan)
dataset.fillna(dataset.median(), inplace = True)
dataset[279] = dataset[279]-1

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
ensemble_num = 4
batch_size = 16
num_workers = 0
num_epochs = 20
hidden_dims = 279
num_categories = 16

params = {'batch_size': batch_size,
          'num_workers': num_workers}

df_dataset=pandas_dataset(dataset)
train, test = random_split(df_dataset,[400,52])
train_loader = [DataLoader(train, shuffle=True, **params) for _ in range(ensemble_num)]
test_loader = DataLoader(test,**params)


def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py)


targets = torch.cat([y for x, y in test_loader], dim=0).to(device)
#targets = torch.reshape(targets,(-1,))

'''probs_map = predict(test_loader, arrhythmia_model.to(device))
acc_map = (probs_map.argmax(-1) == targets).float().mean()
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
print(f'[MAP] Acc.: {acc_map:.1%}; NLL: {nll_map:.3}')'''
def optuna_model(trial):
    # Main MIMO model layer
    class MIMOModel(nn.Module):
        def __init__(self, hidden_dim: int = hidden_dims, ensemble_num: int = 3):
            super(MIMOModel, self).__init__()
            self.output_dim = trial.suggest_int('output_dim', 32, 256)
            self.input_layer = nn.Linear(hidden_dim, hidden_dim * ensemble_num)
            self.backbone_model = BackboneModel(hidden_dim, ensemble_num, self.output_dim)
            self.ensemble_num = ensemble_num
            self.output_layer = nn.Linear(self.output_dim, num_categories * ensemble_num)

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

    # Middle layer for the MIMO model that is mainly configured by optuna
    class BackboneModel(nn.Module):

        def __init__(self, hidden_dim: int, ensemble_num: int, output_dim: int):
            super(BackboneModel, self).__init__()
            layers = []
            in_features = hidden_dim * ensemble_num
            num_layers = trial.suggest_int('num_layers', 1, 4)
            for i in range(num_layers):
                out_dim = trial.suggest_int('n_units_l{}'.format(i), 64, 1024)
                layers.append(nn.Linear(in_features, out_dim))
                layers.append(nn.ReLU())
                dropout_rate = trial.suggest_float('dr_rate_l{}'.format(i), 0.0, 0.5)
                if dropout_rate > 0.05:
                    layers.append(nn.Dropout(dropout_rate))
                in_features = out_dim
            layers.append(nn.Linear(in_features, output_dim))
            layers.append(nn.ReLU())
            self.layers = layers

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            module = nn.Sequential(*self.layers).to(device)
            output = module(x)
            return output

    mimo_optuna = MIMOModel(hidden_dim=hidden_dims, ensemble_num=ensemble_num)
    return mimo_optuna

bug_dict = {}


def objective(trial):
    # Model and main parameter initialization

    model = optuna_model(trial=trial).to(device)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, 'Adadelta')(model.parameters(), lr=lr)
    gamma = trial.suggest_float('gamma', 0.1, 0.8)
    scheduler = StepLR(optimizer, step_size=len(train_loader[0]), gamma=gamma)

    # Training and eval loop
    for epoch in range(num_epochs):
        model.train()
        for datum in zip(*train_loader):
            # Training
            model_inputs = torch.stack([data[0] for data in datum]).to(device)
            targets = torch.stack([data[1] for data in datum]).to(device)
            targets = targets.squeeze()
            ensemble_num, batch_size = list(targets.size())
            optimizer.zero_grad()
            outputs = model(model_inputs)
            '''print('Preds Before')
            print(outputs.size())'''
            loss = F.nll_loss(
                outputs.reshape(ensemble_num * batch_size, -1), targets.reshape(ensemble_num * batch_size)
            )
            loss.backward()

            optimizer.step()
            scheduler.step()
        '''print('Target Tensor')
        print(targets.reshape(ensemble_num * batch_size))
        print(targets.reshape(ensemble_num * batch_size).size())
        print('Pred Tensor')
        print(outputs.reshape(ensemble_num * batch_size, -1))
        print(outputs.reshape(ensemble_num * batch_size, -1).size())'''
        # Evaluation

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                model_inputs = torch.stack([data[0]] * ensemble_num).to(device)
                target = data[1].to(device)

                outputs = model(model_inputs)
                #bug_dict['outputs'] = outputs
                output = torch.mean(outputs, axis=1)
                #bug_dict['output'] = output
                #print(output)
                #print(target)
                #bug_dict['target'] = target
                test_loss += F.nll_loss(output, target.squeeze(), reduction="sum").item()
                pred = output.argmax(dim=-1, keepdim=True).flatten()
                correct += pred.eq(target.view_as(pred)).sum().item()
        '''print('Target Tensor')
        print(target)
        print(target.size())
        print('Pred Tensor')
        print(pred)
        print(pred.size())'''
        test_loss /= len(test_loader.dataset)
        acc = 100.0 * correct / len(test_loader.dataset)
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return acc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=60000)

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


class Config(NamedTuple):
    """
    Hyperparameters
    """

    #: random seed
    seed = 42
    # training epochs
    num_epochs = 100
    # batch size
    batch_size = batch_size
    #: learning rate
    learning_rate = 0.05
    #: learning rate step gamma
    gamma = 0.1
    #: num workers
    num_workers = num_workers

    train_log_interval = 50
    valid_log_interval = 100

    """
    MIMO Hyperparameters
    """
    ensemble_num = ensemble_num


'''
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
train_dataloaders = [
    DataLoader(train_dataset, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=True)
    for _ in range(Config.ensemble_num)
]
'''
configure = Config
targets = []
preds = []
data_load = []

test_input = None
test_target = None
'''class MIMOTrainer:
    def __init__(
            self,
            config: Config,
            model: nn.Module,
            train_dataloaders: List[DataLoader],
            test_dataloader: DataLoader,
            device: torch.device,
    ):
        self.config = config
        self.model = model
        self.train_dataloaders: List[DataLoader] = train_dataloaders
        self.test_dataloader: DataLoader = test_dataloader

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.device = device

    def train(self):
        self.model.to(self.device)
        self.model.train()
        global_step = 0
        for epoch in range(1, self.config.num_epochs + 1):
            for datum in zip(*self.train_dataloaders):
                model_inputs = torch.stack([data[0] for data in datum]).to(self.device)
                model_inputs.reshape(self.config.ensemble_num, self.config.batch_size,1,-1)
                targets = torch.stack([data[1] for data in datum]).to(self.device)

                if len(data_load) == 0:
                    data_load.append(model_inputs)
                ensemble_num, batch_size = list(targets.size())
                self.optimizer.zero_grad()
                outputs = self.model(model_inputs)
                loss = F.cross_entropy(
                    outputs.reshape(ensemble_num * batch_size, -1), targets.reshape(ensemble_num * batch_size)
                )
                # print(loss)
                loss.backward()

                self.optimizer.step()

                global_step += 1
                if global_step != 0 and global_step % self.config.train_log_interval == 0:
                    print(f"[Train] epoch:{epoch} \t global step:{global_step} \t loss:{loss:.4f}")
                if global_step != 0 and global_step % self.config.valid_log_interval == 0:
                    self.validate()

    def validate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in self.test_dataloader:
                model_inputs = torch.stack([data[0]] * self.config.ensemble_num).to(self.device)
                target = data[1].to(self.device)
                targets.append(data[1])

                outputs = self.model(model_inputs)
                output = torch.mean(outputs, axis=1)

                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=-1, keepdim=True)
                preds.append(pred.flatten().data.tolist())
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_dataloader.dataset)
        acc = 100.0 * correct / len(self.test_dataloader.dataset)
        print(f"[Valid] Average loss: {test_loss:.4f} \t Accuracy:{acc:2.2f}%")
        self.model.train()'''


class MIMOTrainer:
    def __init__(
            self,
            config: Config,
            model: nn.Module,
            train_dataloaders: List[DataLoader],
            test_dataloader: DataLoader,
            device: torch.device,
    ):
        self.config = config
        self.model = model
        self.train_dataloaders: List[DataLoader] = train_dataloaders
        self.test_dataloader: DataLoader = test_dataloader

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=len(self.train_dataloaders[0]), gamma=config.gamma)

        self.device = device

    def train(self):
        self.model.to(self.device)
        self.model.train()
        global_step = 0
        for epoch in range(1, self.config.num_epochs + 1):
            for datum in zip(*self.train_dataloaders):
                model_inputs = torch.stack([data[0] for data in datum]).to(self.device)
                targets = torch.stack([data[1] for data in datum]).to(self.device)

                ensemble_num, batch_size = list(targets.size())
                self.optimizer.zero_grad()
                outputs = self.model(model_inputs)
                loss = F.nll_loss(
                    outputs.reshape(ensemble_num * batch_size, -1), targets.reshape(ensemble_num * batch_size)
                )
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                global_step += 1
                if global_step != 0 and global_step % self.config.train_log_interval == 0:
                    print(f"[Train] epoch:{epoch} \t global step:{global_step} \t loss:{loss:.4f}")
                if global_step != 0 and global_step % self.config.valid_log_interval == 0:
                    self.validate()

    def validate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in self.test_dataloader:
                model_inputs = torch.stack([data[0]] * self.config.ensemble_num).to(self.device)
                target = data[1].to(self.device)

                outputs = self.model(model_inputs)
                output = torch.mean(outputs, axis=1)

                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=-1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_dataloader.dataset)
        acc = 100.0 * correct / len(self.test_dataloader.dataset)
        print(f"[Valid] Average loss: {test_loss:.4f} \t Accuracy:{acc:2.2f}%")
        self.model.train()

# model = MIMOModel(hidden_dim=hidden_dims, ensemble_num=ensemble_num)
# trainer = MIMOTrainer(configure, model, train_loader, test_loader, device)
# trainer.train()
