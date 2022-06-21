import torch
from laplace import Laplace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import pickle
#import helper
import logging
from config import casting_args, models

#Laplace approximation for casting dataset

casting_train_path = casting_args['train_set_path']
casting_test_path = casting_args['test_set_path']

casting_model_path = models['base_models']['casting']

with open(casting_model_path, "rb") as fin:
    casting_model = pickle.load(fin)

batch_size = 128
image_resolution = 127
num_workers = 0


transformations = transforms.Compose([transforms.Resize(int((image_resolution+1)/2)*3),
                                      transforms.RandomCrop(image_resolution),
                                      transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])

train_set = datasets.ImageFolder(casting_train_path, transform=transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True
                                           , num_workers= num_workers, pin_memory=True)

test_set = datasets.ImageFolder(casting_test_path, transform=transformations)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers= num_workers)

la = Laplace(casting_model,likelihood='classification')



