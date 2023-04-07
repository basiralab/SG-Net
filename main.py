"""
Main function of StairwayGraphNet framework for inter- and intra-modality brain graph resolution and synthesis.

Details can be found in:
(1) the original paper https://link.springer.com/chapter/10.1007/978-3-030-87589-3_15
    Islem Mhiri,  Mohamed Ali Mahjoub, and Islem Rekik. "Non-isomorphic Inter-modality Graph Alignment and Synthesis for Holistic Brain Mapping", MICCAI 2020, Lima, Peru.
(2) the youtube channel of BASIRA Lab:
---------------------------------------------------------------------

This file contains the implementation of three main steps of our StairwayGraphNet framework:
  (1) brain graph alignment,
  (2) brain graph prediction, and
  (3) brain graph super-resolution

  StairwayGraphNet(X_train_source, X_test_source, X_train_target1, X_test_target1, X_train_target2, X_test_target2)
          Inputs:
                  X_train_source:   training source brain graphs
                  X_test_source:    testing source brain graphs
                  X_train_target1:   training target 1 brain graphs
                  X_test_target1:    testing target 1 brain graphs
                  X_train_target2:   training target 2 brain graphs
                  X_test_target2:    testing target 2 brain graphs
          Output:
                  predicted_graph:  A list of size (m × n1× n1 ) stacking the predicted brain graphs where m is the number of subjects and n1 is the number of regions of interest
                  data_target: A list of size (m × n1× n1 ) stacking the target brain graphs where m is the number of subjects and n1 is the number of regions of interest
                  source_test: A list of size (m × n× n ) stacking the source brain graphs where m is the number of subjects and n is the number of regions of interest
                  l1_test: the MAE between the predicted and target brain graphs
                  eigenvector_test: The MAE between the predicted and target eigenvector centralities

To evaluate our framework we used 3 fold-CV stratefy.

---------------------------------------------------------------------
Copyright 2021 Islem Mhiri, Sousse University.
Please cite the above paper if you use this code.
All rights reserved.
"""



import os.path as osp
import pickle
from scipy.linalg import sqrtm
import numpy
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv, BatchNorm
import argparse
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from torch.distributions import normal, kl
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, InnerProductDecoder, ARGVA
from torch_geometric.utils import train_test_split_edges
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold
from losses import*
from model import*
from preprocess import*
from prediction import*
from centrality import *
from plots import*
warnings.filterwarnings("ignore")


"""#Training"""

torch.cuda.empty_cache()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")

source_data = np.random.normal(0, 0.5, (150, 595))
target_data1 = np.random.normal(0, 0.5, (150, 12720))
target_data2 = np.random.normal(0, 0.5, (150, 35778))

kf = KFold(n_splits=3, shuffle=True, random_state=1773)

fold = 0
losses_test = []
closeness_losses_test = []
# betweenness_losses_test = []
eigenvector_losses_test = []

for train_index, test_index in kf.split(source_data):
    # print( * "#" + " FOLD " + str(fold) + " " +  * "#")
    X_train_source, X_test_source, X_train_target1, X_test_target1, X_train_target2, X_test_target2 = source_data[train_index], source_data[test_index], target_data1[train_index], target_data1[test_index], target_data2[train_index], target_data2[test_index]

    source_test, predicted_test1, data_target1, l1_test1, eigenvector_test1, predicted_test2, data_target2, l1_test2, eigenvector_test2 = StairwayGraphNet(X_train_source, X_test_source, X_train_target1, X_test_target1, X_train_target2, X_test_target2)



test_mean1 = np.mean(l1_test1)
test_mean2 = np.mean(l1_test2)
Eigenvector_test_mean1 = np.mean(eigenvector_test1)
Eigenvector_test_mean2 = np.mean(eigenvector_test2)
plot_source(source_test)
plot_target1(data_target1)
plot_target1(predicted_test1)
plot_target2(data_target2)
plot_target2(predicted_test2)

print("Mean L1 Test 1", test_mean1)

print("Mean Eigenvector Test 1", Eigenvector_test_mean1)

print("Mean L1 Test 2", test_mean2)

print("Mean Eigenvector Test 2", Eigenvector_test_mean2)

