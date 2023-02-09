import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from rostok.graph_grammar.node import ROOT, GraphGrammar
from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary

node_vocab = NodeVocabulary()
node_vocab.add_node(ROOT)
node_vocab.create_node("J")
node_vocab.create_node("L")
node_vocab.create_node("F")
node_vocab.create_node("M")
node_vocab.create_node("EF")
node_vocab.create_node("EM")

rule_vocab = RuleVocabulary(node_vocab)

rule_vocab.create_rule("FlatCreate", ["ROOT"], ["F"], 0, 0)
rule_vocab.create_rule("Mount", ["F"], ["F", "M", "EM"], 0, 0, [(0, 1), (1, 2)])
rule_vocab.create_rule("MountAdd", ["M"], ["M", "EM"], 0, 1, [(0, 1)])
rule_vocab.create_rule("FingerUpper", ["EM"], ["J", "L", "EM"], 0, 2, [(0, 1), (1, 2)])

graph = GraphGrammar()
graph_states = []
rule_actions = []

NUMBER = 5
fig1, axs1 = plt.subplots(1, NUMBER)

# Randomly selects rules and shows their impact on the graph.
for i in range(NUMBER):
    list_rules = rule_vocab.get_list_of_applicable_rules(graph)
    rand_rule_name = np.random.choice(list_rules)
    rand_rule = rule_vocab.get_rule(rand_rule_name)

    graph.apply_rule(rand_rule)

    # Show graphs
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph},
                     ax=axs1[i])

    axs1[i].set(xlabel="Rule action: " + rand_rule_name)

plt.show()
