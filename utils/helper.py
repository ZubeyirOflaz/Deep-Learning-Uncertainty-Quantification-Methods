import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import pandas as pd
import optuna
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torchviz import make_dot, make_dot_from_trace
import pickle


class MimoTrainValidateCasting:
    def __init__(self, mimo_model, hyperparameter_dict: dict, trainloader, testloader):
        super(MimoTrainValidateCasting, self).__init__()
        self.model = mimo_model
        self.dict = hyperparameter_dict
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.train_loader = trainloader
        self.test_loader = testloader
        self.ensemble_num = len(trainloader)
        print('init complete')

    def model_train(self, num_epochs=50):
        train_loader = self.train_loader
        device = self.device
        model = self.model.to(device)
        lr = self.dict['lr']
        gamma = self.dict['gamma']
        optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=(len(train_loader[0])), gamma=gamma)
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
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
            print(f'{epoch}: {train_loss}')
            self.model_validate()
        self.model = model
        print('model has been updated')
        return model, self

    def model_validate(self, get_predictions=False):
        test_loader = self.test_loader
        device = self.device
        ensemble_num = self.ensemble_num
        model = self.model.to(device)
        model.eval()
        test_loss = 0
        test_size = 0
        correct = 0
        predictions = []
        targets = []
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
                if get_predictions is True:
                    predictions.append(outputs.cpu().tolist())
                    targets.append(target.cpu().tolist())
                # target_test = target.view_as(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= test_size
        acc = 100.0 * correct / test_size
        print(f'Current Accuracy: {acc}')
        if get_predictions is True:
            return predictions, targets

class MimoTrainValidateFordA:
    def __init__(self, mimo_model, hyperparameter_dict: dict, trainloader, testloader):
        super(MimoTrainValidateFordA, self).__init__()
        self.model = mimo_model
        self.dict = hyperparameter_dict
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.train_loader = trainloader
        self.test_loader = testloader
        self.ensemble_num = len(trainloader)
        print('init complete')

    def model_train(self, num_epochs=50):
        ensemble_num = self.ensemble_num
        train_loader = self.train_loader
        device = self.device
        model = self.model.to(device)
        lr = self.dict['lr'].values[0]
        gamma = self.dict['gamma'].values[0]
        optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=(len(train_loader[0])), gamma=gamma)
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
            self.model_validate()
        self.model = model
        print('model has been updated')
        return model, self

    def model_validate(self, get_predictions=False):
        test_loader = self.test_loader
        device = self.device
        ensemble_num = self.ensemble_num
        model = self.model.to(device)
        model.eval()
        test_loss = 0
        test_size = 0
        correct = 0
        predictions = []
        targets = []
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
                if get_predictions is True:
                    predictions.extend(output.cpu().tolist())
                    targets.extend(target.cpu().tolist())
                # target_test = target.view_as(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= test_size
        acc = 100.0 * correct / test_size
        print(f'Current Accuracy: {acc}')
        if get_predictions is True:
            return predictions, targets

def create_study_analysis(optuna_study):
    parameters = [i.params for i in optuna_study]
    accuracy = [y.value for y in optuna_study]
    state = [i.state.name for i in optuna_study]
    df = pd.DataFrame(parameters)
    df.insert(0, 'accuracy', accuracy)
    df.assign(trial_state=state, inplace=True)
    df.sort_values('accuracy', ascending=False, inplace=True)
    return df


def load_mimo_model(study_number: int, model_number : int, model_dict = None):
    from casting_mimo_model import mimo_cnn_model
    from utils.ford_a_optuna_models import optuna_ford_a_mimo
    with open(f'model_repo\\study_{study_number}.pkl', 'rb') as fin:
        study = pickle.load(fin)
    study_df = create_study_analysis(study.get_trials())
    if model_dict is None:
        model = mimo_cnn_model(study.get_trials()[model_number])
    else:
        model = optuna_ford_a_mimo(study.get_trials()[model_number], model_dict)
    state = model.load_state_dict(torch.load(f'model_repo\\{model_number}_{study_number}.pyt'))
    print(state)
    study_dict = study_df.loc[model_number]
    return model, study_dict




def weighted_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def weighted_classes_arrhythmia(a_dataset, n_classes=16, return_count = False):
    count = [0] * n_classes
    for _, y in enumerate(a_dataset):
        count[y[1].numpy()[0]] += 1
    weight_per_class = [0.] * n_classes
    N = float(sum(count))
    for i in range(n_classes):
        if count[i] == 0:
            weight_per_class[i] = 0
        else:
            weight_per_class[i] = N / float(count[i])
    weight = [0] * len(a_dataset)
    for idx, val in enumerate(a_dataset):
        weight[idx] = weight_per_class[val[1]]
    if return_count:
        return weight, count
    else:
        return weight


def plot_network(data_loader, model=None, model_path=None, from_path=True):
    if from_path:
        with open(model_path, 'rb') as fin:
            model = pickle.load(fin)
    x, y = next(iter(data_loader))
    model_graph = make_dot(y, params=dict(list(model.named_parameters())))





def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                             'Trouser',
                             'Pullover',
                             'Dress',
                             'Coat',
                             'Sandal',
                             'Shirt',
                             'Sneaker',
                             'Bag',
                             'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
