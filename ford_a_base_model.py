import os
from config import dataset_paths as args
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ford_a_utils import ARFFDataset
from utils.ford_a_utils import FordAConvModel
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['ford_a_train']
test_set_path = ROOT_DIR + args['ford_a_test']

batch_size = 16
num_workers = 0

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

train_dataset = ARFFDataset(train_set_path)
test_dataset = ARFFDataset(test_set_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)


def ford_a_train():
    model = FordAConvModel()
    lr = 5e-4
    gamma = 0.9
    optimizer = getattr(optim, 'Adadelta')(model.parameters(), lr=lr)
    num_epoch = 50
    scheduler = StepLR(optimizer, step_size=len(train_loader), gamma=gamma)
    for i in range(num_epoch):
        t0 = time.time()
        # logging.error('training has started')
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'Seconds between epoch:{time.time() - t0}')
        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)
        print(accuracy)
    return model
