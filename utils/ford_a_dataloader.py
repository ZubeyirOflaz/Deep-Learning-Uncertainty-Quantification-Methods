import torch
import numpy as np
import pandas
from scipy.io import arff


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