import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
import numpy as np
from torch_geometric.data import Data
from torch.autograd import Variable




class Aligner(torch.nn.Module):
    def __init__(self):
        super(Aligner, self).__init__()

        nn = Sequential(Linear(1, 1225), ReLU())
        self.conv1 = NNConv(35, 35, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, 35), ReLU())
        self.conv2 = NNConv(35, 1, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, 35), ReLU())
        self.conv3 = NNConv(1, 35, nn, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3 = torch.cat([F.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr))), x1], dim=1)
        x4 = x3[:, 0:35]
        x5 = x3[:, 35:70]

        x6 = (x4 + x5) / 2
        return x6








class Generator1(nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()

        nn = Sequential(Linear(1, 1225),ReLU())
        self.conv1 = NNConv(35, 35, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, 5600), ReLU())
        self.conv2 = NNConv(160, 35, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, 5600), ReLU())
        self.conv3 = NNConv(35, 160, nn, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(160, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


        # self.layer= torch.nn.ConvTranspose2d(160, 160,5)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        # x = torch.squeeze(x)

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        # x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        # x2 = F.dropout(x2, training=self.training)

        x3 = F.sigmoid(self.conv33(self.conv3(x1, edge_index, edge_attr)))
        x3 = F.dropout(x3, training=self.training)



        x4  = torch.matmul(x3.t(), x3)

        return x4

class Discriminator1(torch.nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        self.conv1 = GCNConv(160, 160, cached=True)
        self.conv2 = GCNConv(160, 1, cached=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        x = torch.squeeze(x)
        x1 = F.sigmoid(self.conv1(x, edge_index))
        x1 = F.dropout(x1, training=self.training)
        x2 = F.sigmoid(self.conv2(x1, edge_index))
        #         # x2 = F.dropout(x2, training=self.training)


        return x2

class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()

        
        self.conv21 = GCNConv(160, 2 * 268, cached=True)
        self.conv211 = BatchNorm(2 * 268, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)
        self.conv22 = GCNConv(2 * 268, 268, cached=True)
        self.conv222 = BatchNorm(268, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)
        self.conv23 = GCNConv(268, 268, cached=True)
      

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        x = torch.squeeze(x)
        x = self.conv21(x, edge_index).relu()
        x1 = F.sigmoid(self.conv211(x))
        x1 = F.dropout(x1, training=self.training)
       
        x2 = self.conv22(x1, edge_index).relu()
        x2 = F.sigmoid(self.conv222(x2))
        x2 = F.dropout(x2, training=self.training)
       
        x3  = (torch.matmul(x2.t(), x2)) 

        return x3


class Discriminator2(torch.nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        
        self.conv21 = GCNConv(268, 268, cached=True)
        self.conv211 = BatchNorm(268, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)
        self.conv22 = GCNConv(268, 1, cached=True)
        self.conv222 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x = torch.squeeze(x)
        x = self.conv21(x, edge_index).relu()
        x = self.conv211(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x1 = F.relu(self.conv222(self.conv22(x, edge_index)))

        return F.sigmoid(x1)
        

