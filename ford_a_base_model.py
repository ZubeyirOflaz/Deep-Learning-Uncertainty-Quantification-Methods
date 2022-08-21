import os
from config import dataset_paths as args
import torch
import torch.nn.functional as F
from utils.ford_a_dataloader import ARFFDataset
from utils.ford_a_utils import FordAConvModel
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time
import torcheck
import logging
import optuna
from utils.ford_a_optuna_models import optuna_ford_a
import pickle
import random
from optuna.trial import TrialState
from utils.helper import create_study_analysis

study_name = str(random.randint(7000000, 7999999))


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['ford_a_train']
test_set_path = ROOT_DIR + args['ford_a_test']

batch_size = 16
num_workers = 0

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#torch.backends.cudnn.benchmark = True

train_dataset = ARFFDataset(train_set_path, data_scaling = True)
test_dataset = ARFFDataset(test_set_path, data_scaling = True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


def objective(trial):
    # Generate the model.
    try:
        model = optuna_ford_a(trial).to(device)
    except Exception as e:
        print('Infeasible model, trial will be skipped')
        print(e)
        raise optuna.exceptions.TrialPruned()

    logging.debug('model has been compiled')
    num_epoch = 75
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    gamma = trial.suggest_float('gamma', 0.8, 1)
    scheduler = StepLR(optimizer, step_size=(len(train_loader)), gamma=gamma)

    # Training of the model.
    for epoch in range(num_epoch):
        t0 = time.time()
        logging.debug('training has started')
        t_loss = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), \
                           target.type(torch.LongTensor).to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data.unsqueeze(dim = 1))
            loss = F.nll_loss(output, target.flatten())
            t_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'Seconds between epoch:{time.time() - t0}')
        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # Limiting validation data.
                data, target = data.to(device).to(device, non_blocking=True)\
                    , target.to(device, non_blocking=True)
                output = model(data.unsqueeze(dim = 1))
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)
        print(f'Epoch {epoch} accuracy: {accuracy}')
        print(f'Epoch {epoch} loss: {t_loss}')
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    with open(f"model_repo\\{trial.number}_{study_name}.pickle", "wb") as fout:
        pickle.dump(model, fout)

        # Handle pruning based on the intermediate value.

    return accuracy




if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=optuna.samplers.TPESampler(n_startup_trials=80))
    study.optimize(objective, n_trials=250, timeout=60000)

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

with open(f"model_repo\\{study.best_trial.number}_{study_name}.pickle", "rb") as fin:
    best_result = pickle.load(fin)
with open(f'model_repo\\best_models\\ford_a_{study.best_trial.value}.pickle', 'wb') as fout:
    pickle.dump(best_result, fout)

trial_dataframe = create_study_analysis(study.get_trials(deepcopy=True))
with open(f'model_repo\\study_{study.study_name}.pkl', 'wb') as fout:
    pickle.dump(study, fout)

study1 = 7447174