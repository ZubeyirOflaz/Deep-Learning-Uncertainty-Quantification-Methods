import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import pickle
#import helper
import logging
import time
import os

import optuna
from optuna.trial import TrialState
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from config import dataset_paths as args
import random
# Data Import

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['casting_train']
test_set_path = ROOT_DIR + args['casting_test']

batch_size = 128
image_resolution = 127
num_workers = 0

LOG_INTERVAL = 10
EPOCHS = 25

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

random_seed = 15

torch.random.manual_seed(random_seed)
random.seed(random_seed)
transformations = transforms.Compose([transforms.Resize(int((image_resolution+1) * 1.40)),
                                      transforms.RandomCrop(image_resolution),
                                      transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

train_set = datasets.ImageFolder(train_set_path, transform=transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers= num_workers, pin_memory=True)
# images, labels = next(iter(train_loader))
# helper.imshow(images[0], normalize=False)

torch.random.manual_seed(random_seed)
random.seed(random_seed)

test_set = datasets.ImageFolder(test_set_path, transform=transformations)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers= num_workers)

N_TRAIN_EXAMPLES = len(train_set)
N_TEST_EXAMPLES = len(test_set)


def optuna_model(trial):
    layers = []
    num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 1, 3)
    num_dense_nodes = trial.suggest_categorical('num_dense_nodes',
                                                [64, 128, 512, 1024])
    dense_nodes_divisor = trial.suggest_categorical('dense_nodes_divisor',
                                                    [2, 4, 8])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
    drop_out = trial.suggest_discrete_uniform('drop_out', 0.05, 0.5, 0.05)

    dict_params = {'num_cnn_blocks': num_cnn_blocks,
                   'num_dense_nodes': num_dense_nodes,
                   'dense_nodes_divisor': dense_nodes_divisor,
                   'batch_size': batch_size,
                   'drop_out': drop_out}
    input_channels = 1
    for i in range(dict_params['num_cnn_blocks']):
        num_filters = trial.suggest_categorical('num_filters', [16, 32, 48, 64])
        kernel_size = trial.suggest_int('kernel_size', 2, 4)
        # if len(layers) == 0:
        #    input_channels = 1
        # else:
        #    input_channels = layers[-2].out_channels
        layers.append(nn.Conv2d(input_channels, num_filters, kernel_size))
        input_channels = num_filters
        layers.append(nn.MaxPool2d(4, stride=2))
    layers.append(nn.Conv2d(input_channels, 64, kernel_size))
    layers.append(nn.AdaptiveMaxPool2d((12, 12)))
    layers.append(nn.Flatten())
    linear_input = 64 * 12 * 12
    layers.append(nn.Linear(linear_input, dict_params['num_dense_nodes']))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dict_params['drop_out']))
    layers.append(nn.Linear(dict_params['num_dense_nodes'],
                            int(dict_params['num_dense_nodes'] / dict_params['dense_nodes_divisor'])))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dict_params['drop_out']))
    layers.append(nn.Linear(int(dict_params['num_dense_nodes'] / dict_params['dense_nodes_divisor']), 2))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)


def objective(trial):
    # Generate the model.
    model = optuna_model(trial).to(device)
    #print(model)
    logging.error('model has been compiled')

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    for epoch in range(EPOCHS):
        t0 = time.time()
        #logging.error('training has started')
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            #logging.error('forward pass has been completed')
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print(f'Seconds between epoch:{time.time()-t0}')
        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # Limiting validation data.
                if batch_idx * batch_size >= N_TEST_EXAMPLES:
                    break
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(test_loader.dataset), N_TEST_EXAMPLES)
        print(accuracy)

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    with open("model_repo\\{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)

        # Handle pruning based on the intermediate value.


    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25, timeout=60000)

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

with open("model_repo\\{}.pickle".format(study.best_trial.number), "rb") as fin:
    best_result = pickle.load(fin)
with open(f'model_repo\\best_models\\{study.best_trial.value}.pickle', 'wb') as fout:
    pickle.dump(best_result, fout)