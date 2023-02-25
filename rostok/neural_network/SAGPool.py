from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch

class SAGPoolToAlphaZero(torch.nn.Module):
    def __init__(self,args):
        super(SAGPoolToAlphaZero, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_rules = args.num_rules
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio, GNN=GCNConv)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio, GNN=GCNConv)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio, GNN=GCNConv)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.fc_bn1 = torch.nn.BatchNorm1d(self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.fc_bn2 = torch.nn.BatchNorm1d(self.nhid//2)

        self.fc_to_pi = torch.nn.Linear(self.nhid//2, self.num_rules)
        self.fc_to_v = torch.nn.Linear(self.nhid//2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, __, batch, __, __ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, __, batch, __, __ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, __, batch, __, __ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        # x = self.fc_bn1(x)
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        # x = self.fc_bn2(x)
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        pi = F.log_softmax(self.fc_to_pi(x), dim=1)
        v = F.relu(self.fc_to_v(x))

        return pi, v