import networkx as nx
from dataclasses import dataclass
from networkx.algorithms.traversal import dfs_preorder_nodes


class BlockWrapper:
    def __init__(self, block_cls, *args, **kwargs):
        self.builder = None
        self.block_cls = block_cls
        self.args = args
        self.kwargs = kwargs

    def create_block(self):
        if self.builder is None:
            raise Exception('Set builder first')
        return self.block_cls(self.builder, *self.args, **self.kwargs)


@dataclass
class Node:
    label: str = "*"
    is_terminal: bool = False

    # None for non-terminal nodes
    block_wrapper: BlockWrapper = None

    def __hash__(self) -> int:
        return hash(str(self.label) + str(self.is_terminal))

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            raise Exception(
                "Wrong type of comparable object. Must be Node instead {wrong_type}".format(wrong_type=type(__o)))
        return self.label == __o.label


class Rule:
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
            [raw_node["Node"].is_terminal
             for _, raw_node in graph.nodes(data=True)]
        )
        self._graph_insert = graph

    @property
    def is_terminal(self):
        return self._is_terminal

    def __hash__(self):
        return hash(self.graph_insert + hash(self.replaced_node))


@dataclass
class WrapperTuple:
    id: int
    block_wrapper: BlockWrapper  # Set default value


ROOT = Node("ROOT")


class GraphGrammar(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.__uniq_id_counter = -1
        self.add_node(self._get_uniq_id(), Node=ROOT)

    def _get_uniq_id(self):
        self.__uniq_id_counter += 1
        return self.__uniq_id_counter

    def find_nodes(self, match: Node):
        match_nodes = []
        for raw_node in self.nodes.items():
            # Extract node info
            node: Node = raw_node[1]["Node"]
            node_id = raw_node[0]
            if node.label == match.label:
                match_nodes.append(node_id)
        return match_nodes

    def _replace_node(self, node_id, rule: Rule):
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
        rule_graph_relabeled = nx.relabel_nodes(
            rule.graph_insert, relabel_in_rule)

        # Push changes into graph
        self.remove_node(node_id)
        self.add_nodes_from(rule_graph_relabeled.nodes.items())
        self.add_edges_from(rule_graph_relabeled.edges)
        self.add_edges_from(in_edges)
        self.add_edges_from(out_edges)

    def closest_node_to_root(self, list_ids):
        root_id = self.get_root_id()

        def sort_by_root_distance(node_id):
            return len(nx.shortest_path(self, root_id, node_id))

        sorta = sorted(list_ids, key=sort_by_root_distance)
        return sorta[0]

    def get_root_id(self):
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
            out_edges_ids_node = list(
                filter(lambda x: x[0] == id_closest, edge_list))
            if out_edges_ids_node:
                raise Exception("Trying delete not leaf node")
            self.remove_node(id_closest)
        else:
            self._replace_node(id_closest, rule)

    def graph_partition_dfs(self):
        paths = []
        path = []

        dfs_edges = nx.dfs_edges(self)
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
        paths = self.graph_partition_dfs()
        wrapper_array = []

        for path in paths:
            wrapper = []
            for node_id in path:
                node: Node = self.nodes[node_id]["Node"]
                if node.is_terminal:
                    buf = WrapperTuple(node_id, node.block_wrapper)
                    wrapper.append(buf)
                else:
                    raise Exception('Graph contain non-terminal elements')
            wrapper_array.append(wrapper.copy())

        return wrapper_array
