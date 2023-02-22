import os
import sys
import time

import numpy as np
from tqdm import tqdm
from rostok.graph_generators.graph_game import GraphGrammarGame
from rostok.graph_grammar.node import GraphGrammar

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from rostok.neural_network.SAGPool import SAGPoolToAlphaZero as graph_nnet
from rostok.neural_network.converter import ConverterToPytorchGeometric


args_train = dotdict({
    "epochs": 10,
    "batch_size": 1,
    "cuda": torch.cuda.is_available(),
})

class AlphaZeroWrapper(NeuralNet):
    def __init__(self, game: GraphGrammarGame):
        self.action_size = game.getActionSize()
        self.converter = ConverterToPytorchGeometric(game.rule_vocabulary.node_vocab)
        args_network = dotdict({"num_features":len(self.converter.label2id),
                        "num_rules": game.getActionSize(),
                        "nhid": 2,
                        "pooling_ratio": 0.3,
                        "dropout_ratio": 0.3})
        
        self.nnet = graph_nnet(args_network)
        

        if args_train.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (graph, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args_train.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args_train.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args_train.batch_size)
                graph, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                data_graph = self.converter.transform_digraph(graph)
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args_train.cuda:
                    data_graph, target_pis, target_vs = data_graph.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(data_graph)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), data_graph.size(0))
                v_losses.update(l_v.item(), data_graph.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, graph: GraphGrammar):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        data_graph = self.converter.transform_digraph(graph)
        if args_train.cuda: data_graph = data_graph.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(data_graph)

        print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

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
        map_location = None if args_train.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
