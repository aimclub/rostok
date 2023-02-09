import matplotlib.pyplot as plt
import networkx as nx
from rostok.graph_grammar.node import GraphGrammar, Node


def replace_nodes(graph: GraphGrammar, mapping: dict[Node, Node]):
    graph_dict = dict(graph.nodes.items())
    for node_replace in mapping.keys():
        id_replace = graph.find_nodes(node_replace)
        for id in id_replace:
            graph_dict[id]["Node"] = mapping[node_replace]

def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     font_size=8,
                     pos=nx.spring_layout(graph, dim=2),
                     node_size=200,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()

def plot_graph_reward(graph: GraphGrammar, reward):
    plt.figure()
    plt.title('reward of the graph: ' + str(reward))
    nx.draw_networkx(graph,
                     font_size=10,
                     pos=nx.spring_layout(graph, dim=2),
                     node_size=600,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()

def save_graph_plot_reward(graph: GraphGrammar, reward, path):
    plt.figure()
    plt.title('reward of the graph: ' + str(reward))
    nx.draw_networkx(graph,
                     font_size=10,
                     pos=nx.spring_layout(graph, dim=2),
                     node_size=600,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.savefig(path)
