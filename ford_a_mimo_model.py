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
from utils.helper import create_study_analysis

study_name = str(random.randint(8000000, 8999999))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
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
              'device': device,
              'study_name': study_name}

train_dataset = ARFFDataset(train_set_path, data_scaling=False)
test_dataset = ARFFDataset(test_set_path, data_scaling=False)
train_loader = [DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers
                           , pin_memory=True) for _ in range(ensemble_num)]
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
train_dataset = None
test_dataset = None
las_memory_info = {}


def objective(trial):
    # Model and main parameter initialization
    global las_memory_info
    torch.cuda.empty_cache()
    gc.collect()
    # las_memory_info = torch.cuda.memory_stats()
    model = optuna_ford_a_mimo(trial=trial, trial_parameters=model_dict).to(device)
    print(model)
    # torch.cuda.memory_allocated()
    # torch.cuda.memory_reserved()
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-6, 5e-2, log=True)
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
    gamma = trial.suggest_float('gamma', 0.9, 1)
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
            model_inputs = torch.cat([data[0] for data in datum], dim=1).to(device, non_blocking=True)
            # model_inputs = model_inputs[:, :, None, :]
            # bug_dict['model_inputs'] = model_inputs
            targets = torch.stack([data[1] for data in datum]).type(torch.LongTensor).to(device, non_blocking=True)
            # bug_dict['t_targets'] = targets
            # bug_dict['t_targets_a'] = targets
            optimizer.zero_grad()
            outputs = model(model_inputs[:, None, :])
            batch_size = list(targets.size())[1]
            # bug_dict['t_outputs'] = outputs

            loss = F.nll_loss(
                outputs.reshape(ensemble_num * batch_size, -1), targets.reshape(ensemble_num * batch_size)
            )
            train_loss += loss.item()
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
                model_inputs = torch.cat([data[0]] * ensemble_num, dim=1).to(device)
                # model_inputs = model_inputs[:, :, None, :]
                target = data[1].squeeze().type(torch.LongTensor).to(device)
                # bug_dict['data'] = data
                outputs = model(model_inputs[:, None, :])
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
        if trial.should_prune():  # or (epoch > 4 and acc < 52):
            raise optuna.exceptions.TrialPruned()
    torch.save(model.state_dict(), f"model_repo\\{trial.number}_{study_name}.pyt")

    '''with open("model_repo\\{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)'''

    return acc


study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=50),
                            direction='maximize', study_name=study_name)

study.optimize(objective, n_trials=150)

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

best_model = optuna_ford_a_mimo(study.get_trials(deepcopy=True)[study.best_trial.number], model_dict)
best_model.load_state_dict(torch.load(f"model_repo\\{study.best_trial.number}_{study_name}.pyt"))
torch.save(best_model.state_dict(), f'model_repo\\best_models\\ford_a_mimo_{study.best_trial.value}_{study_name}.pyt')

trial_dataframe = create_study_analysis(study.get_trials(deepcopy=True))
with open(f'model_repo\\study_{study.study_name}.pkl', 'wb') as fout:
    pickle.dump(study, fout)

study1 = 8966978
