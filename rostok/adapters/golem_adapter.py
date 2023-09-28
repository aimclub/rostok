import random
from copy import deepcopy
from random import choice
from typing import Any, Dict, Iterable, Optional
from golem.core.dag.graph_node import GraphNode
import networkx as nx
import numpy as np
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory

from rostok.graph_grammar.node import ROOT, GraphGrammar, Node


class GraphGrammarFactory(DefaultOptNodeFactory):

    def __init__(self,
                 available_node_types: Optional[Iterable[str]] = None,
                 num_node_types: Optional[int] = None):
        super().__init__(available_node_types, num_node_types)

    def get_node(self, **kwargs) -> OptNode:
        chosen_node_type = choice(self.available_nodes) \
            if self.available_nodes \
            else random.randint(0, self._num_node_types)

        return OptNode(content={"Node": chosen_node_type, "name": chosen_node_type.label})


class GraphGrammarAdapter(BaseNetworkxAdapter):

    def __init__(self):
        super().__init__()
        self.domain_graph_class = GraphGrammar

    def _node_adapt(self, data: Dict) -> OptNode:
        if not "name" in data:
            data["name"] = data["Node"].label

        return OptNode(content=deepcopy(data))

    def _adapt(self, adaptee: GraphGrammar) -> OptGraph:
        adaptee_copy = deepcopy(adaptee)
        adaptee_copy = adaptee_copy.reverse(copy=False)
        return super()._adapt(adaptee_copy)
    
    def _node_restore(self, node: GraphNode) -> Dict:
        """Transforms GraphNode to dict of NetworkX node attributes.
        Override for custom behavior."""
        if hasattr(node, 'content'):
            return deepcopy(node.content)
        else:
            return {}
    
    def _restore(self,
                 opt_graph: OptGraph,
                 metadata: Optional[Dict[str, Any]] = None) -> GraphGrammar:

        opt_graph_copy = deepcopy(opt_graph)
        nx_adapt_graph = super()._restore(opt_graph_copy, metadata)
        nx_adapt_graph = nx_adapt_graph.reverse(copy=False)
        graph = GraphGrammar()
        # Remove start node
        unused_root = graph.find_nodes(ROOT)
        graph.remove_node(unused_root[0])

        graph.add_nodes_from(nx_adapt_graph.nodes(data=True))
        graph.add_edges_from(nx_adapt_graph.edges(data=True))
        relabel = {golem_id: graph.get_uniq_id() for golem_id in list(graph.nodes)}
        graph = nx.relabel_nodes(graph, relabel, copy=False)
        return graph

    def adapt_node_seq(self, list_node: list[Node]) -> list[OptNode]:
        return [self._node_adapt({"Node": node_data}) \
        for node_data in list_node]
