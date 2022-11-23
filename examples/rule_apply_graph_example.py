import networkx as nx
import matplotlib.pyplot as plt

from rostok.graph_grammar.node import Rule, Node, GraphGrammar
from example_ruleset import FlatCreate, Mount, MountAdd, FingerUpper


G = GraphGrammar()
rule_action = [FlatCreate, Mount, MountAdd, MountAdd, FingerUpper]

for i in rule_action:
    G.apply_rule(i)

plt.figure()
nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800,
                 labels={n: G.get_node_by_id(n).label for n in G})

plt.show()
