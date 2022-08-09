import torch
import numpy as np
import pandas as pd
from scipy.io import arff
import torch.nn as nn
import pandas
import torch.nn.functional as F

class ARFFDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, transforms=None):
        data, meta = arff.loadarff(file_path)
        self.dataframe = pandas.DataFrame(data)

        self.labels = self.dataframe.target
        self.features = self.dataframe.loc[:, self.dataframe.columns != 'target']
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        features = self.features.loc[index].values
        label = int(self.labels.loc[index])

        if label == -1:
            label = 0

        if self.transforms is not None:
            features = np.reshape(features, (25, 20))
            features = self.transforms(features)
        else:
            features = torch.Tensor(features)

        label_tensor = torch.Tensor([label])

        return features, label_tensor


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


class FordAMIMOModel(nn.Module):

    def __init__(self, ensemble_size, intermediate_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.pool = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, dilation=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        # The intermediate_dim depends on ensemble_size
        self.fc1 = nn.Linear(intermediate_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, ensemble_size)

    def forward(self, x):
        # In my implementation: batch_size = ensemble_size
        # Input: (batch_size, num_features)
        x = x.reshape((1, -1))
        # Shape: (1, batch_size * num_features)
        x = torch.unsqueeze(x, dim=1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No softmax -> every unit is one individual predictor
        x = torch.sigmoid(self.fc3(x))
        x = torch.squeeze(x, dim=0)
        return x
