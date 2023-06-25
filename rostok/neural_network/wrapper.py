import os
import sys
import time
from typing import Union

import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm

sys.path.append('../../')

import torch
import torch.optim as optim

class NNWraper:
    def __init__(self, nnet, args):
        self.args = args
        self.nnet = nnet.to(args["device"])
        
        self.optimizer = optim.Adam(self.nnet.parameters())
        self.mse = torch.nn.MSELoss()
        
        self.loss_history = []
        

    def update(self, data_loader: DataLoader):
        """
        examples: list of examples, each example is of form (graph, pi, v)
        """

        self.nnet.train()
        temp_loss = []
        for x in data_loader:
            batch_size, num_nodes = x.size()
            
            x = x.to(self.args["device"])
            # compute output
            values = self.nnet(x)

            loss = self.mse(values, x.y)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            temp_loss.append(loss.detach().cpu().numpy())
        self.loss_history.append(np.mean(temp_loss))
        return np.mean(temp_loss)

    def predict(self, data_graph: Union[Data, DataLoader]):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        data_graph = data_graph
        self.nnet.eval()
        if isinstance(data_graph, Data):
            with torch.no_grad():
                value = self.nnet(data_graph)
        if isinstance(data_graph, DataLoader):
            with torch.no_grad():
                for data in data_graph:
                    value = self.nnet(data)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return value.detach()

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location=self.args["device"])
        self.nnet.load_state_dict(checkpoint['state_dict'])
