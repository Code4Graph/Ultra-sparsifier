import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, adj, nfeat, nhid, nclass, dropout, device, adj_update):
        super(GCN, self).__init__()
        #1: delete, 0: keep
        self.device = device
        self.gc1 = GraphConvolution(nfeat, nhid, adj, adj_update)
        self.gc2 = GraphConvolution(nhid, nclass, adj, adj_update)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)
