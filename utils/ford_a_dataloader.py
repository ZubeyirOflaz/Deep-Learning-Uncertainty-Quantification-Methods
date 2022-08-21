import torch
import numpy as np
import pandas
from scipy.io import arff
import torch.nn as nn
import os
from config import dataset_paths as args


class ARFFDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, transforms=None, data_scaling=False):
        data, meta = arff.loadarff(file_path)
        self.dataframe = pandas.DataFrame(data)

        self.labels = self.dataframe.target
        self.features = self.dataframe.loc[:, self.dataframe.columns != 'target']
        self.transforms = transforms
        self.data_scaling = data_scaling
        if data_scaling:
            from sklearn import preprocessing
            features = self.features.values
            self.transformer = preprocessing.RobustScaler().fit(features)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        features = self.features.loc[index].values
        label = int(self.labels.loc[index])

        if label == -1:
            label = 0
        if self.data_scaling:
            features = self.transformer.transform(features.reshape(1, -1))
            features = torch.Tensor(features).squeeze(0)
        elif self.transforms is not None:
            features = np.reshape(features, (25, 20))
            features = self.transforms(features)
        else:
            features = torch.Tensor(features)
        label_tensor = torch.Tensor([label])

        return features, label_tensor


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    train_set_path = ROOT_DIR + args['ford_a_train']
    test_set_path = ROOT_DIR + args['ford_a_test']

    train_dataset_scaled = ARFFDataset(train_set_path, data_scaling = True)
    train_dataset_not_scaled = ARFFDataset(train_set_path, data_scaling = False)

