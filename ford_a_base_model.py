import os
from config import dataset_paths as args
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ford_a_utils import ARFFDataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['forda_train']
test_set_path = ROOT_DIR + args['forda_test']#


class FordAConvModel(nn.Module):

    def __init__(self, intermediate_dim=512):
        # Intermediate dim:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.pool = nn.MaxPool1d(3)

        self.drop = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(intermediate_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.conv4(x))
        x = self.drop(x)
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc3(x))

        x = torch.squeeze(x, dim=1)
        return x
