import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from laplace import Laplace


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device='cuda')
            y = y.to(device='cuda')

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')


def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()


def calculate_metric_base(model: Laplace, dataloader: DataLoader, n_trials=20):
    means = []
    standard_deviations = []
    predictions = []
    targets = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    for (x, y) in dataloader:
        input = x.to(device)
        output = y.to(device)
        pred = model.predictive_samples(x=input, n_samples=n_trials, pred_type='nn')
        x_m = pred.mean(axis=0)
        means.append(x_m)
        x_std = pred.std(axis=0)
        standard_deviations.append(x_std)
        x_h = x_m.argmax(dim=1, keepdim = True).squeeze()
        predictions.append(x_h)
        targets.append(output)
    result_dict = {}
    predictions2 = torch.cat(predictions)
    targets2 = torch.cat(targets)
    result_dict['means'] = torch.cat(means)
    result_dict['standard_deviations'] = torch.cat(standard_deviations)
    result_dict['predictions'] = predictions2
    result_dict['targets'] = targets2
    result_dict['accuracy'] = predictions2.eq(targets2.view_as(predictions2))
    return result_dict


def calculate_metric_mimo(model: torch.nn.Module, dataloader: DataLoader, ensemble_num):
    predictions = []
    outputs = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    for data in dataloader:
        model_inputs = torch.cat([data[0]] * ensemble_num, dim=3).to(device)
        pred = model(model_inputs)
        output = torch.mean(pred, axis=1)
        predictions.append(pred)
        outputs.append(output)
    predictions = torch.cat(predictions)
    return predictions

def calculate_kullback_leibner(model : torch.nn.Module, dataloader: DataLoader, ensemble_num):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    for (x,y) in dataloader:
        input = x.to(device)
        output = y.to(device)
        pred = model(input)


