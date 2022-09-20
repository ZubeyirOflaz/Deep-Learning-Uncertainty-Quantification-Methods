import os
from config import dataset_paths as args
import torch
import torch.nn.functional as F
from utils.ford_a_dataloader import ARFFDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time
import logging
import optuna
from utils.ford_a_optuna_models import optuna_ford_a_experimental
import pickle
import random
from optuna.trial import TrialState
from utils.helper import create_study_analysis
from thop import profile

study_name = str(random.randint(6000000, 6999999))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['ford_a_train']
test_set_path = ROOT_DIR + args['ford_a_test']

batch_size = 16
num_workers = 0

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

train_dataset = ARFFDataset(train_set_path, data_scaling=True)
test_dataset = ARFFDataset(test_set_path, data_scaling=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def objective(trial):
    # Generate the model.
    try:
        model = optuna_ford_a_experimental(trial).to(device)
    except Exception as e:
        print('Infeasible model, trial will be skipped')
        print(e)
        raise optuna.exceptions.TrialPruned()

    logging.debug('model has been compiled')
    num_epoch = 50
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    gamma = trial.suggest_float('gamma', 0.95, 1)
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
            output = model(data.unsqueeze(dim=1))
            loss = F.nll_loss(output, target.flatten())
            t_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device, non_blocking=True) \
                    , target.to(device, non_blocking=True)
                output = model(data.unsqueeze(dim=1))
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)
        print(f'Epoch {epoch} accuracy: {accuracy}, loss: {t_loss}')
        # trial.report(accuracy, epoch)
    flops, prms = profile(model, inputs=(data.unsqueeze(dim=1),), verbose=False)

    with open(f"model_repo\\{trial.number}_{study_name}.pickle", "wb") as fout:
        pickle.dump(model, fout)


    return (accuracy, flops)


if __name__ == "__main__":
    study = optuna.create_study(study_name=study_name,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=80, multivariate=True),
                                directions=['maximize', 'minimize'])
    study.optimize(objective, n_trials=200, timeout=60000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))


test = study.trials_dataframe()
with open(f'model_repo\\study_{study.study_name}.pkl', 'wb') as fout:
    pickle.dump(study, fout)



def ford_a_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.tensor(target).type(torch.LongTensor)
    data = torch.stack(data).unsqueeze(1)
    return [data, target]

train_dataset = ARFFDataset(train_set_path, data_scaling = True)
test_dataset = ARFFDataset(test_set_path, data_scaling = True)
train_loader_c = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=num_workers, pin_memory= True, collate_fn= ford_a_collate, shuffle=False)
test_loader_c = DataLoader(test_dataset, batch_size=batch_size,
                         num_workers=num_workers, pin_memory= True, collate_fn= ford_a_collate, shuffle=False)
from utils.evaluation_metrics import predict,get_metrics

with open(f"model_repo\\{88}_{study_name}.pickle", "rb") as fin:
    chosen_model = pickle.load(fin)

preds = predict(train_loader_c, chosen_model).to(device)
preds_test = predict(test_loader_c, chosen_model).to(device)
targets_train = torch.cat([y for x, y in train_loader_c], dim=0).cpu()
targets_train = torch.reshape(targets_train,(-1,)).to(device)
targets_test = torch.cat([y for x, y in test_loader_c], dim=0).cpu()
targets_test = torch.reshape(targets_test,(-1,)).to(device)
get_metrics(preds, targets_train)
get_metrics(preds_test, targets_test)
p_input, yy = next(iter(train_loader_c))
flops, prms = profile(chosen_model, inputs = (p_input.to(device),),verbose=False)

visual = optuna.visualization.plot_pareto_front(study, target_names=["total_loss", "acc"])
visual_2 = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="acc"
)
visual_3 = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="total_loss"
)

study1 = 6660793
