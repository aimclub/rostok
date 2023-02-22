import networkx as nx
import numpy as np

import torch
from torch_geometric.data import Data

from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.graph_grammar.node import GraphGrammar


class ConverterToPytorchGeometric:
    def __init__(self, node_vocalubary: NodeVocabulary):

        self.label2node = node_vocalubary.node_dict
        self.id2label, self.label2id = self._create_dict_node_labels(node_vocalubary)

    def _create_dict_node_labels(
            self, node_vocabulary: NodeVocabulary) -> tuple[dict[int, str], dict[str, int]]:

        sorted_node_labels = sorted(node_vocabulary.node_dict.keys())

        dict_id_label_nodes = dict(enumerate(sorted_node_labels))
        dict_label_id_nodes = {label: idx for (idx, label) in enumerate(sorted_node_labels)}

        return dict_id_label_nodes, dict_label_id_nodes

    def flatting_sorted_graph(self, graph: GraphGrammar) -> tuple[list[int], list[list[int]]]:

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

    def transform_digraph(self, graph: GraphGrammar):

        node_label_list, edge_id_list = self.flatting_sorted_graph(graph)

        one_hot_list = list(map(self.one_hot_encodding, node_label_list))

        edge_index = torch.t(torch.tensor(edge_id_list, dtype=torch.long))
        x = torch.tensor(one_hot_list, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)

        return data