from typing import Union
import networkx as nx
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.graph_grammar.node import GraphGrammar


class TorchAdapter:

    def __init__(self, node_vocabulary: NodeVocabulary):

        self.label2node = node_vocabulary.node_dict

        self.label2id, self.id2label = self._create_dict_node_labels(node_vocabulary)

    def _create_dict_node_labels(
            self, node_vocabulary: NodeVocabulary) -> tuple[dict[int, str], dict[str, int]]:

        sorted_node_labels = sorted(node_vocabulary.node_dict.keys(),
                                    key=lambda x: hash(node_vocabulary.node_dict[x]))

        dict_id_label_nodes = dict(enumerate(sorted_node_labels))
        dict_label_id_nodes = {label: idx for (idx, label) in enumerate(sorted_node_labels)}

        return dict_id_label_nodes, dict_label_id_nodes

    def flating_graph(self, graph: GraphGrammar) -> tuple[list[str], list[list[int]]]:

        sorted_id_nodes = list(
            nx.lexicographical_topological_sort(graph, key=lambda x: graph.get_node_by_id(x).label))
        sorted_name_nodes = list(map(lambda x: graph.get_node_by_id(x).label, sorted_id_nodes))

        id_node2list = {id[1]: id[0] for id in enumerate(sorted_id_nodes)}

        list_edges_on_id = list(graph.edges)
        list_id_edge_links = list(map(lambda x: [id_node2list[n] for n in x], list_edges_on_id))

        if not list_id_edge_links:
            list_id_edge_links = [[0, 0]]

        return sorted_name_nodes, list_id_edge_links

    def one_hot_encodding(self, label_node: str) -> list[int]:

        one_hot = np.zeros(len(self.label2id), dtype=int)
        one_hot[self.label2id[label_node]] = 1
        return one_hot.tolist()

    def gg2torch_data(self, graph: GraphGrammar, y=None):

        node_label_list, edge_id_list = self.flating_graph(graph)

        one_hot_list = list(map(self.one_hot_encodding, node_label_list))

        edge_index = torch.t(torch.Tensor(edge_id_list, dtype=torch.int32))
        x = torch.Tensor(one_hot_list, dtype=torch.float32)

        if y:
            y = torch.Tensor(y, dtype=torch.float32)
            data = Data(x=x, edge_index=edge_index, y=y)
        else:
            data = Data(x=x, edge_index=edge_index)

        return data

    def list_gg2data_loader(self, graphs: list[GraphGrammar], batch_size, targets=None):
        data_list: list[Data] = []
        for graph, y in zip(graphs, targets):
            data_list.append(self.gg2torch_data(graph, y))

        return DataLoader(data_list, batch_size=batch_size)