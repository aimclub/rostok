import torch
import torch.nn.functional as F


import torch_geometric.utils as pyg_utils
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import sys
from math import ceil
torch.set_printoptions(threshold=sys.maxsize)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None
        self.double()

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()

        x0 = x
        #IPython.embed()
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self, args, variational=False):
        super(Net, self).__init__()

        self.variational = variational
        

        self.args = args
        self.max_nodes = args["max_nodes"]
        num_channels = args["num_features"]
        
        num_nodes = ceil(0.25 * self.max_nodes)
        self.gnn1_pool = GNN(num_channels, 64, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(num_channels, 64, 64, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)

        out_channels = 2 if self.variational else 1

        self.lin2 = torch.nn.Linear(64, out_channels)
        self.double()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        adj = pyg_utils.to_dense_adj(edge_index, batch).to(self.args["device"]).double()
        x, mask = pyg_utils.to_dense_batch(x, batch)
        x = x.to(self.args["device"]).double()
        mask = mask.to(self.args["device"])
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        #print('time for gnn2')

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))

        x = self.lin2(x)

        return x#, l1 + l2, e1 + e2
