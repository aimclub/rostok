from rostok.graph_grammar.node import GraphGrammar, Node


def replace_nodes(graph: GraphGrammar, mapping: dict[Node, Node]):
    graph_dict = dict(graph.nodes.items())
    for node_replace in mapping.keys():
        id_replace = graph.find_nodes(node_replace)
        for id in id_replace:
            graph_dict[id]["Node"] = mapping[node_replace]
