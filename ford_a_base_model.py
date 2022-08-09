import os
from config import dataset_paths as args
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ford_a_utils import ARFFDataset
from utils.ford_a_utils import FordAConvModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = ROOT_DIR + args['ford_a_train']
test_set_path = ROOT_DIR + args['ford_a_test']

train_dataset = ARFFDataset(train_set_path)
test_dataset = ARFFDataset(test_set_path)


