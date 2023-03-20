import networkx as nx

from rostok.graph.graph import Graph
from rostok.graph.node import Node


class Rule:
    """ The class contains a graph object for substitution into the generated graph
    and the target node which will be replaced by :py:attr:`Rule.graph_insert`.
    The feature of the rule's terminality is automatically determined.
    Id's mean V from graph theory, do not intersect with V from generated graph.
    """
    _graph_insert: nx.DiGraph = nx.DiGraph()
    replaced_node: Node = Node()
    # In local is system!
    id_node_connect_child = -1
    id_node_connect_parent = -1
    _is_terminal: bool = None

    @property
    def graph_insert(self):
        return self._graph_insert

    @graph_insert.setter
    def graph_insert(self, graph: nx.DiGraph):
        self._is_terminal = all(
            [raw_node["Node"].is_terminal for _, raw_node in graph.nodes(data=True)])
        self._graph_insert = graph

    @property
    def is_terminal(self):
        return self._is_terminal

    def __hash__(self):
        return hash(str(self.graph_insert) + str(self.replaced_node))


class GraphGrammar(Graph):
    """ A class for using generative rules (similar to L grammar) and
        manipulating the construction graph.
        The mechanism for assignment a unique Id, each added node using :py:meth:`GraphGrammar.rule_apply`
        will increase the counter.
        Supports methods from :py:class:`networkx.DiGraph` ancestor class
    """

    def __init__(self, rule_vocablary = None,**attr):
        super().__init__(**attr)
        self.rule_vocabulary = rule_vocablary
        self.rule_list = []
        self.counter_nonterminal_rules = 0

    def _replace_node(self, node_id: int, rule: Rule):
        """Auxiliary function for apply rule

        Args:
            node_id (int):
            rule (Rule):
        """

        # Convert to list for mutable
        in_edges = [list(edge) for edge in self.in_edges(node_id)]
        out_edges = [list(edge) for edge in self.out_edges(node_id)]

        id_node_connect_child_graph = self._get_uniq_id()
        # FIXME: Why is it inverted? 
        is_equal_id = rule.id_node_connect_parent != rule.id_node_connect_child
        id_node_connect_parent_graph = self._get_uniq_id(
        ) if is_equal_id else id_node_connect_child_graph

        relabel_in_rule = \
            {rule.id_node_connect_child: id_node_connect_child_graph,
            rule.id_node_connect_parent: id_node_connect_parent_graph}

        for raw_nodes in rule.graph_insert.nodes.items():
            raw_node_id = raw_nodes[0]
            if raw_node_id in relabel_in_rule.keys():
                continue
            relabel_in_rule[raw_node_id] = self._get_uniq_id()

        for edge in in_edges:
            edge[1] = id_node_connect_parent_graph
        for edge in out_edges:
            edge[0] = id_node_connect_child_graph

        # Convert ids in rule to graph ids system
        rule_graph_relabeled = nx.relabel_nodes(rule.graph_insert, relabel_in_rule)

        # Push changes into graph
        self.remove_node(node_id)
        self.add_nodes_from(rule_graph_relabeled.nodes.items())
        self.add_edges_from(rule_graph_relabeled.edges)
        self.add_edges_from(in_edges)
        self.add_edges_from(out_edges)

    def apply_rule(self, rule: Rule):
        if isinstance(rule, Rule):
            self.rule_list.append("Rule_Object")
        else:
            self.rule_list.append(rule)
            rule = self.rule_vocabulary.get_rule(rule)
            if self.rule_vocabulary is None:
                raise Exception("Without rule vocabulary the rules must be Rule class objects.")

        if not rule.is_terminal:
            self.counter_nonterminal_rules += 1

        ids = self.find_nodes(rule.replaced_node)
        edge_list = list(self.edges)
        id_closest = self.closest_node_to_root(ids)
        if rule.graph_insert.order() == 0:
            # Stub removing leaf node if input rule is empty
            out_edges_ids_node = list(filter(lambda x: x[0] == id_closest, edge_list))
            if out_edges_ids_node:
                raise Exception("Trying delete not leaf node")
            self.remove_node(id_closest)
        else:
            self._replace_node(id_closest, rule)
