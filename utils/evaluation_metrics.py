import math
import numpy as np
import torch
from torch.utils.data import DataLoader
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

def calculate_metric_base(model : torch.nn.Module, dataloader : DataLoader):
    predictions = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    for x, _ in dataloader:
        pred = model(x.to(device))
        predictions.append(pred)
    predictions = torch.cat(predictions)
    '''standard_dev = np.std(predictions)
    mean = np.mean(predictions)
    prediction = np.round(predictions)'''
    return predictions

def calculate_metric_mimo(model : torch.nn.Module, dataloader : DataLoader, ensemble_num):
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
