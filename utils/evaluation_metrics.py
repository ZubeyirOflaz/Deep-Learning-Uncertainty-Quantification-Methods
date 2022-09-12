import math

import numpy
import numpy as np
import pandas
import torch
from torch.utils.data import DataLoader
from laplace import Laplace
import pandas as pd
import torch.distributions as dists
import torchmetrics


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
            py.append(torch.softmax(model(x.cuda()), dim=-1).cpu().detach().numpy())
    if not laplace:
        return torch.Tensor(numpy.concatenate(py)).cpu()
    return torch.cat(py).cpu()


def get_metrics(predictions, targets, ece_bins=10):
    accuracy = (predictions.argmax(-1) == targets).float().mean()
    nll = -dists.Categorical(predictions).log_prob(targets).mean()
    ece = torchmetrics.functional.calibration_error(predictions, targets, n_bins=ece_bins)
    rmsce = torchmetrics.functional.calibration_error(predictions, targets, n_bins=ece_bins, norm='l2')
    print(f'Acc: {accuracy}, nll: {nll}, ece: {ece}, rmsce: {rmsce}')


def calculate_ensemble_divergence(tensor: torch.Tensor):
    divergence = torch.zeros(tensor.size()[0], 1)
    for i in range(tensor.size()[0]):
        divergence[i] = torch.sub(tensor, tensor[i]).abs().sum()
    return divergence.transpose(0, 1)


def create_kullback_leibner_divergence(tensor: torch.Tensor):
    t_mean = tensor.mean(axis=1)
    divergence_matrix = torch.zeros(tensor.size())
    divergence_amount = torch.zeros(list(tensor.size())[:2])
    for i in range(tensor.size()[0]):
        divergence_matrix[i] = torch.sub(tensor[i], t_mean[i])
        divergence_amount[i] = calculate_ensemble_divergence(divergence_matrix[i])
    return divergence_matrix, divergence_amount


def calculate_metric_laplace(model: Laplace, dataloader: DataLoader, n_trials=50):
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
        x_h = x_m.argmax(dim=1, keepdim=True).squeeze()
        predictions.append(x_h)
        targets.append(output)
    result_dict = {}
    try:
        predictions2 = torch.cat(predictions)
    except:
        predictions2[len(predictions) - 1] = predictions[len(predictions) - 1].unsqueeze(dim=0)
        predictions2 = torch.cat(predictions)
    targets2 = torch.cat(targets)
    result_dict['means'] = torch.cat(means).cpu().detach().numpy()
    result_dict['standard_deviations'] = torch.cat(standard_deviations).cpu().detach().numpy()
    result_dict['predictions'] = predictions2.cpu().detach().numpy()
    result_dict['targets'] = targets2.cpu().detach().numpy()
    result_dict['accuracy'] = predictions2.eq(targets2.view_as(predictions2)).cpu().detach().numpy()
    return result_dict


def calculate_metric_mimo(model: torch.nn.Module, dataloader: DataLoader, ensemble_num: int, model_type: str):
    predictions = []
    means = []
    standard_deviations = []
    outputs_master = []
    targets = []
    k_l_matrices = []
    k_l_values = []
    result_dict = {}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.eval()
    if model_type == 'ford_a':
        concat_dim = 1
    else:
        concat_dim = 3
    for data in dataloader:
        model_inputs = torch.cat([data[0]] * ensemble_num, dim=concat_dim).to(device)
        target = data[1].squeeze().type(torch.LongTensor).to(device)
        if model_type == 'ford_a':
            outputs = model(model_inputs[:, None, :])
        else:
            outputs = model(model_inputs).cpu().detach()
        output = torch.mean(outputs, axis=1).cpu().detach()
        standard_deviation = torch.std(outputs, axis=1).cpu().detach()
        prediction = output.argmax(dim=-1, keepdim=True)
        k_l_matrix, k_l_value = create_kullback_leibner_divergence(outputs)
        predictions.append(outputs)
        means.append(output)
        standard_deviations.append(standard_deviation)
        outputs_master.append(prediction)
        targets.append(target)
        k_l_matrices.append(k_l_matrix.cpu().detach())
        k_l_values.append(k_l_value.cpu().detach())
    prediction = torch.cat(outputs_master)
    try:
        targets = torch.cat(targets).cpu().detach()
    except:
        targets[len(targets) - 1] = targets[len(targets) - 1].unsqueeze(dim=0)
        targets = torch.cat(targets).cpu().detach()
    result_dict['means'] = torch.cat(means).cpu().detach().numpy()
    result_dict['standard_deviations'] = torch.cat(standard_deviations).cpu().detach().numpy()
    result_dict['kullback_leibner_matrices'] = torch.cat(k_l_matrices).cpu().detach().numpy()
    result_dict['kullback_leibner_values'] = torch.cat(k_l_values).cpu().detach().numpy()
    result_dict['kullback_leibner_sum'] = np.sum(result_dict['kullback_leibner_values'], axis=1)
    result_dict['predictions'] = prediction.cpu().detach().numpy()
    result_dict['targets'] = targets.cpu().detach().numpy()
    result_dict['accuracy'] = prediction.eq(targets.view_as(prediction)).cpu().detach().numpy()
    return result_dict


def create_metric_dataframe(metrics_dict, mimo_metric=False):
    num_class = metrics_dict['means'].shape[1]
    columns = []
    if mimo_metric:
        num_ensemble = metrics_dict['kullback_leibner_values'].shape[1]
    columns.extend([f'm_{i}' for i in range(num_class)])
    columns.extend([f'std_{i}' for i in range(num_class)])
    if mimo_metric:
        columns.extend(([f'ensemble_divergece_{i}' for i in range(num_ensemble)]))
        columns.extend(['total_divergence'])
    columns.extend(['predicted', 'actual', 'accuracy'])
    metrics_numpy = np.concatenate([metrics_dict['means'], metrics_dict['standard_deviations']], axis=1)
    if mimo_metric:
        metrics_numpy = np.concatenate([metrics_numpy, metrics_dict['kullback_leibner_values']], axis=1)
        # metrics_numpy = np.concatenate([metrics_numpy, ], axis=1)
    if mimo_metric:
        metrics_numpy2 = np.stack([metrics_dict['kullback_leibner_sum'][:, np.newaxis], metrics_dict['predictions'],
                                   metrics_dict['targets'][:, np.newaxis], metrics_dict['accuracy']]).transpose()
    else:
        metrics_numpy2 = np.stack([metrics_dict['predictions'], metrics_dict['targets'],
                                   metrics_dict['accuracy']]).transpose()
    metrics_numpy = np.concatenate([metrics_numpy, metrics_numpy2.squeeze()], axis=1)
    df = pd.DataFrame(metrics_numpy, columns=columns)
    df['accuracy'] = df['accuracy'].astype('bool')
    return df


def get_very_high_confidence(dataframe: pd.DataFrame, column_name: str, ascending=True):
    dataset_len = len(dataframe)
    dataset_filtered = dataframe[dataframe['dataset'] == 'train']
    dataset_filtered = dataset_filtered[dataset_filtered['accuracy'] == False]
    dataset_filtered.sort_values(column_name, ascending=ascending, inplace=True)
    dataset_filtered.reset_index(inplace=True)
    num_mistakes_allowed = int(dataset_len / 100)
    absolute_confidence = dataset_filtered.loc[num_mistakes_allowed][column_name]
    high_confidence = dataset_filtered.loc[num_mistakes_allowed][column_name]
    return absolute_confidence, high_confidence
