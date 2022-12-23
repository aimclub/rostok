from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import pychrono as chrono
from networkx.algorithms.traversal import dfs_preorder_nodes


class BlockWrapper:
    """Class is interface between node and interpretation in simulation.

    The interface allows you to create an interpretation of terminal nodes in the simulation.
    Interpretation classes is in :py:mod:`node_render`.
    The instance must be specified when creating the node.
    When assembling a robot from a graph, an object is created by the :py:meth:`BlockWrapper.create_block` method.
    When the object is created, the desired arguments of the interpretation object are set.

    Args:
        block_cls: Interpretation class of node in simulation
        args: Arguments py:attr:`BlockWrapper.block_cls`
        kwargs: Additional arguments py:attr:`BlockWrapper.block_cls`
    """

    def __init__(self, block_cls, *args, **kwargs):
        self.block_cls = block_cls
        self.args = args
        self.kwargs = kwargs

    def create_block(self, builder):
        return self.block_cls(builder, *self.args, **self.kwargs)


@dataclass
class Node:
    """Contains information about the label and :py:class:`BlockWrapper`,
    which is the physical representation of the node in the simulator
    """
    label: str = "*"
    is_terminal: bool = False

    # None for non-terminal nodes
    block_wrapper: BlockWrapper = None

    def __hash__(self) -> int:
        return hash(str(self.label) + str(self.is_terminal))

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            raise Exception(
                "Wrong type of comparable object. Must be Node instead {wrong_type}".format(
                    wrong_type=type(__o)))
        return self.label == __o.label


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


@dataclass
class WrapperTuple:
    """ The return type is used to build the Robot.
        Id - from the generated graph
    """
    id: int
    block_wrapper: BlockWrapper  # Set default value


ROOT = Node("ROOT")


class GraphGrammar(nx.DiGraph):
    """ A class for using generative rules (similar to L grammar) and
        manipulating the construction graph.
        The mechanism for assignment a unique Id, each added node using :py:meth:`GraphGrammar.rule_apply`
        will increase the counter.
        Supports methods from :py:class:`networkx.DiGraph` ancestor class
    """

    def __init__(self, **attr):
        super().__init__(**attr)
        self.__uniq_id_counter = -1
        self.add_node(self._get_uniq_id(), Node=ROOT)

    def _get_uniq_id(self):
        self.__uniq_id_counter += 1
        return self.__uniq_id_counter

    def find_nodes(self, match: Node) -> list[int]:
        """

        Args:
            match (Node): Node for find, matched by label

        Returns:
            list[int]: Id of matched nodes
        """

        match_nodes = []
        for raw_node in self.nodes.items():
            # Extract node info
            node: Node = raw_node[1]["Node"]
            node_id = raw_node[0]
            if node.label == match.label:
                match_nodes.append(node_id)
        return match_nodes

    def _replace_node(self, node_id: int, rule: Rule):
        """Applies rules to node_id

        Args:
            node_id (int):
            rule (Rule):
        """

        # Convert to list for mutable
        in_edges = [list(edge) for edge in self.in_edges(node_id)]
        out_edges = [list(edge) for edge in self.out_edges(node_id)]

        id_node_connect_child_graph = self._get_uniq_id()

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

    def closest_node_to_root(self, list_ids: list[int]) -> int:
        """Find closest node to root from list_ids

        Args:
            list_ids (list[int]):

        Returns:
            int: id of closest Node
        """

        root_id = self.get_root_id()

        def sort_by_root_distance(node_id):
            return len(nx.shortest_path(self, root_id, node_id))

        sorta = sorted(list_ids, key=sort_by_root_distance)
        return sorta[0]

    def get_root_id(self) -> int:
        """

        Returns:
            int: root id
        """

        for raw_node in self.nodes.items():
            raw_node_id = raw_node[0]
            if self.in_degree(raw_node_id) == 0:
                root_id = raw_node_id
        return root_id

    def apply_rule(self, rule: Rule):
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

    def node_levels_bfs(self) -> list[list[int]]:
        """Devide nodes into levels.

        Return a list of lists of nodes where each inner list is a
        level in respect to the \'root\', which is the node with no in edges.
        This function should be reviewed once we start to use graphs with cycles and not just trees
        """
        levels = []
        # Get the root node that has no in_edges. Currently, we assume
        # that where is only one node without in_edges
        for raw_node in self.nodes.items():
            raw_node_id = raw_node[0]
            if self.in_degree(raw_node_id) == 0:
                root_id = raw_node_id

        current_level = [root_id]
        next_level = []
        # The list of edges that is built on the bases of the range to the source
        bfs_edges_list = list(nx.bfs_edges(self, source=root_id))
        for edge in bfs_edges_list:
            # If the edge starts in current level, the end of the edge appends to the next level
            if edge[0] in current_level:
                next_level.append(edge[1])
            # Else, the current level is finished and appended to the levels. next_level becomes
            # current_level and the end of the edge goes to the new  next_level
            else:
                levels.append(current_level)
                current_level = next_level
                next_level = [edge[1]]

        # Finish the levels by appending current and next_level. In the cycle the appending occurs,
        # when the edge of the next level is found.
        levels.append(current_level)
        levels.append(next_level)
        return levels

    def graph_partition_dfs(self) -> list[list[int]]:
        """ 2D list
            Row is branch of graph
            Collum is id node

        Returns:
            list[list[int]]:
        """

        paths = []
        path = []

        root_id = self.get_root_id()
        dfs_edges = nx.dfs_edges(self, root_id)
        dfs_edges_list = list(dfs_edges)

        for edge in dfs_edges_list:
            if len(self.out_edges(edge[1])) == 0:
                path.append(edge[1])
                paths.append(path.copy())
                path = []
            else:
                if len(path) == 0:
                    path.append(edge[0])
                    path.append(edge[1])
                else:
                    path.append(edge[1])
        return paths

    def build_terminal_wrapper_array(self) -> list[list[WrapperTuple]]:
        """Returns a 2-d array of the shape dfs_partition

        Raises:
            Exception: Graph contain non-terminal elements

        Returns:
            list[list[WrapperTuple]]:
        """

        paths = self.graph_partition_dfs()
        wrapper_array = []

        for path in paths:
            wrapper = []
            for node_id in path:
                node: Node = self.get_node_by_id(node_id)
                if node.is_terminal:
                    buf = WrapperTuple(node_id, node.block_wrapper)
                    wrapper.append(buf)
                else:
                    raise Exception('Graph contain non-terminal elements')
            wrapper_array.append(wrapper.copy())

        return wrapper_array

    def get_node_by_id(self, node_id: int) -> Node:
        return self.nodes[node_id]["Node"]

    def get_ids_in_dfs_order(self) -> list[int]:
        """Iterate in deep first order over node ids
        One of the options to present the graph in a flat form

        Returns:
            list[int]:
        """

        return list(dfs_preorder_nodes(self, self.get_root_id()))

    def __eq__(self, __o) -> bool:
        if isinstance(__o, GraphGrammar):
            is_node_eq = __o.nodes == self.nodes
            is_edge_eq = __o.edges == self.edges
            return is_edge_eq and is_node_eq
        return False
