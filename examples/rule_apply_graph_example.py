
from context import node
from node import *

J = Node("J")
L = Node("L")
P = Node("P")
U = Node("U")
M = Node("M")
EF = Node("EF")
EM = Node("EM")
ROOT = Node("ROOT")

# Create rules
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
rule_action = [PalmCreate, Mount, MountAdd, MountAdd, MountUpper, FingerUpper]


plt.figure()
nx.draw_networkx(G, pos=nx.planar_layout(G), node_size=500, labels={n: G.nodes[n]["Node"].label for n in G})
for i in rule_action:
    G.apply_rule(i)
plt.figure()
nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800,
                 labels={n: G.nodes[n]["Node"].label for n in G})

plt.show()