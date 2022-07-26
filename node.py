import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass


class BlockType:
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
    block_type = None


@dataclass
class Rule:
    graph_insert: nx.DiGraph = nx.DiGraph()
    replaced_node: Node = Node()
    # In local is system!
    id_node_connect_child = -1
    id_node_connect_parent = -1


ROOT = Node("ROOT")


class Grammar(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.__uniq_id_counter = -1
        self.add_node(self.get_uniq_id(), Node=ROOT)

    def get_uniq_id(self):
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

    def replace_node(self, node_id, rule: Rule):
        # Convert to list for mutable
        in_edges = [list(edge) for edge in self.in_edges(node_id)]
        out_edges = [list(edge) for edge in self.out_edges(node_id)]

        id_node_connect_child_graph = self.get_uniq_id()
        id_node_connect_parent_graph = self.get_uniq_id() \
            if rule.id_node_connect_parent != rule.id_node_connect_child else id_node_connect_child_graph

        relabel_in_rule = {rule.id_node_connect_child: id_node_connect_child_graph,
                           rule.id_node_connect_parent: id_node_connect_parent_graph}

        for raw_nodes in rule.graph_insert.nodes.items():
            raw_node_id = raw_nodes[0]
            if raw_node_id in relabel_in_rule.keys():
                continue
            relabel_in_rule[raw_node_id] = self.get_uniq_id()

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

    def closest_node_to_root(self, list_ids):
        for raw_node in self.nodes.items():
            raw_node_id = raw_node[0]
            if self.in_degree(raw_node_id) == 0:
                root_id = raw_node_id

        def sort_by_root_distance(node_id):
            return len(nx.shortest_path(self, root_id, node_id))

        sorta = sorted(list_ids, key=sort_by_root_distance)
        return sorta[0]

    def apply_rule(self, rule: Rule):
        ids = self.find_nodes(rule.replaced_node)
        id_closest = self.closest_node_to_root(ids)
        self.replace_node(id_closest, rule)

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

def main():
    J = Node("J")
    L = Node("L")
    P = Node("P")
    R = Node("R")
    U = Node("U")
    E = Node("E")
    M = Node("M")

    ABC_RULE1 = Node("ABC_RULE1")
    ABC_RULE2 = Node("ABC_RULE2")

    G = Grammar()

    G.add_node(0, Node=P)
    G.add_node(1, Node=U)
    G.add_node(2, Node=U)
    G.add_node(3, Node=U)
    G.add_node(4, Node=U)

    G.add_node(5, Node=J)
    G.add_node(7, Node=J)
    G.add_node(8, Node=J)

    G.add_node(9, Node=L)
    G.add_node(10, Node=L)
    G.add_node(11, Node=L)

    G.add_node(12, Node=E)
    G.add_node(13, Node=E)

    G.add_node(14, Node=J)
    G.add_node(15, Node=L)
    G.add_node(16, Node=E)

    nx.add_path(G, [0, 1, 2, 3, 4])
    nx.add_path(G, [5, 9, 12])
    nx.add_path(G, [7, 10, 13])
    nx.add_path(G, [8, 11, 14, 15, 16])

    G.add_edges_from([(1, 5), (2, 7), (3, 8), (3, 4)])

    # create a simple rule

    Rula_Palmer = Rule()

    rule_graph = nx.DiGraph()
    rule_graph.add_node(0, Node=ABC_RULE1)
    rule_graph.add_node(1, Node=ABC_RULE2)
    rule_graph.add_edge(0, 1)

    Rula_Palmer.id_node_connect_child = 0
    Rula_Palmer.id_node_connect_parent = 0
    Rula_Palmer.graph_insert = rule_graph

    Rula_Palmer.replaced_node = E

    for i in range(20):
        G.get_uniq_id()

    print(G.closest_node_to_root([16, 4, 5]))
    print(G.find_nodes(J))

    plt.figure()
    nx.draw_networkx(G, pos=nx.planar_layout(G), node_size=500, labels={n: G.nodes[n]["Node"].label for n in G})

    G.apply_rule(Rula_Palmer)
    what = list(nx.dfs_preorder_nodes(G, source=0))
    print(what)
    plt.figure()
    nx.draw_networkx(G, pos=nx.planar_layout(G), node_size=500, labels={n: G.nodes[n]["Node"].label for n in G})

    plt.figure()
    nx.draw_networkx(G, pos=nx.planar_layout(G), node_size=500)

    plt.show()


def main2():
    J = Node("J")
    L = Node("L")
    P = Node("P")
    ROOT = Node("ROOT")
    U = Node("U")

    M = Node("M")

    EF = Node("EF")
    EM = Node("EM")

    PalmCreate = Rule()

    rule_graph = nx.DiGraph()
    rule_graph.add_node(1, Node=P)

    PalmCreate.id_node_connect_child = 0
    PalmCreate.id_node_connect_parent = 0
    PalmCreate.graph_insert = rule_graph

    PalmCreate.replaced_node = ROOT

    Mount = Rule()

    rule_graph = nx.DiGraph()
    rule_graph.add_node(0, Node=P)
    rule_graph.add_node(1, Node=EM)
    rule_graph.add_edge(0, 1)

    Mount.id_node_connect_child = 0
    Mount.id_node_connect_parent = 1
    Mount.graph_insert = rule_graph

    Mount.replaced_node = P

    MountAdd = Rule()

    rule_graph = nx.DiGraph()
    rule_graph.add_node(0, Node=M)
    rule_graph.add_node(1, Node=EM)
    rule_graph.add_edge(0, 1)

    MountAdd.id_node_connect_child = 1
    MountAdd.id_node_connect_parent = 0
    MountAdd.graph_insert = rule_graph

    MountAdd.replaced_node = EM

    MountUpper = Rule()

    rule_graph = nx.DiGraph()
    rule_graph.add_node(0, Node=U)
    rule_graph.add_node(1, Node=EF)
    rule_graph.add_edge(0, 1)

    MountUpper.id_node_connect_child = 0
    MountUpper.id_node_connect_parent = 0
    MountUpper.graph_insert = rule_graph

    MountUpper.replaced_node = M

    FingerUpper = Rule()

    rule_graph = nx.DiGraph()
    rule_graph.add_node(0, Node=J)
    rule_graph.add_node(1, Node=L)
    rule_graph.add_node(2, Node=EF)
    rule_graph.add_edge(0, 1)
    rule_graph.add_edge(1, 2)

    FingerUpper.id_node_connect_child = 2
    FingerUpper.id_node_connect_parent = 0
    FingerUpper.graph_insert = rule_graph

    FingerUpper.replaced_node = EF




    G = Grammar()

    rule_action = [PalmCreate,Mount,MountAdd,MountAdd,MountUpper,FingerUpper]
    plt.figure()
    nx.draw_networkx(G, pos=nx.planar_layout(G), node_size=500, labels={n: G.nodes[n]["Node"].label for n in G})
    for i in rule_action:
        G.apply_rule(i)
    plt.figure()
    nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800, labels={n: G.nodes[n]["Node"].label for n in G})

    plt.show()

def main3():
    J = Node("J")
    L = Node("L")
    P = Node("P")
    R = Node("R")
    U = Node("U")
    E = Node("E")
    M = Node("M")

    II = Node("II", is_terminal=True)

    ABC_RULE1 = Node("ABC_RULE1")
    ABC_RULE2 = Node("ABC_RULE2")

    G = Grammar()

    G.add_node(0, Node=P)
    G.add_node(1, Node=U)
    G.add_node(2, Node=U)
    G.add_node(3, Node=U)
    G.add_node(4, Node=U)

    G.add_node(5, Node=J)
    G.add_node(7, Node=J)
    G.add_node(8, Node=J)

    G.add_node(9, Node=L)
    G.add_node(10, Node=L)
    G.add_node(11, Node=L)

    G.add_node(12, Node=E)
    G.add_node(13, Node=E)

    G.add_node(14, Node=J)
    G.add_node(15, Node=L)
    G.add_node(16, Node=E)

    nx.add_path(G, [0, 1, 2, 3, 4])
    nx.add_path(G, [5, 9, 12])
    nx.add_path(G, [7, 10, 13])
    nx.add_path(G, [8, 11, 14, 15, 16])

    G.add_edges_from([(1, 5), (2, 7), (3, 8), (3, 4)])

    # create a simple rule

    Rula_Palmer = Rule()

    rule_graph = nx.DiGraph()
    rule_graph.add_node(0, Node=ABC_RULE1)
    rule_graph.add_node(1, Node=ABC_RULE2)
    rule_graph.add_edge(0, 1)

    Rula_Palmer.id_node_connect_child = 0
    Rula_Palmer.id_node_connect_parent = 0
    Rula_Palmer.graph_insert = rule_graph

    Rula_Palmer.replaced_node = E

    for i in range(20):
        G.get_uniq_id()


    dfs_edges = nx.dfs_edges(G)
    dfs_edges_list = list(dfs_edges)
    print(dfs_edges)
    for edge in dfs_edges_list:
        print(f'EDGE{edge}')
        print(f'OUT{G.out_edges(edge[1])}')


    paths = []
    path = []

    for edge in dfs_edges_list:

        if len(G.out_edges(edge[1])) == 0:
            path.append(edge[1])
            paths.append(path.copy())
            path = []
        else:
            if len(path) == 0:
                path.append(edge[0])
                path.append(edge[1])
            else:
                path.append(edge[1])
                
    #for pp in paths:
        #print(pp)

    res = G.graph_partition_dfs()
    for i in res:
        print(i)


    plt.figure()
    nx.draw_networkx(G, pos=nx.planar_layout(G), node_size=500, labels={n: G.nodes[n]["Node"].label for n in G})



    plt.figure()
    nx.draw_networkx(G, pos=nx.planar_layout(G), node_size=500)

    plt.show()


if __name__ == '__main__':
    main3()
