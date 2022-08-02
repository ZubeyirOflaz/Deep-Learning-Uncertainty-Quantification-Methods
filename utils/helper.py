import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import pandas as pd
import optuna
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F


class MimoTrainValidate:
    def __init__(self, mimo_model, hyperparameter_dict: dict, trainloader, testloader):
        super(MimoTrainValidate, self).__init__()
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
                    predictions.append(pred.cpu().tolist())
                    targets.append(target.cpu().tolist())
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


def test_network(net, trainloader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.pause(10)
    plt.close()

    return ax


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




