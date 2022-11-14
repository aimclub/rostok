from dataclasses import dataclass
from engine.node import Node, GraphGrammar
from engine.node_vocabulary import NodeVocabulary

# The deeper the node, the greater the reward
def get_graph_sum_complex_reward(graph: GraphGrammar, reward_map: dict[str, float], node_vocab: NodeVocabulary) -> float:
    dfs = graph.graph_partition_dfs()
    sum = 0
    for branch in dfs:
        branch_len = len(branch)
        for number, nodes_id in enumerate(branch):
            node = graph.nodes.get(nodes_id)["Node"]
            try:
                node_reward = reward_map[node.label]
            except KeyError:
                raise Exception(
                    "There is no node labeled : {label} in reward_map".format(label=node.label))
            node_reward = node_reward * (number+1)/(branch_len+1)
            sum += node_reward

    return sum


def get_graph_mul_reward(graph: GraphGrammar, reward_map: dict[str, float]) -> float:
    dfs = graph.graph_partition_dfs()
    mul = 1
    for branch in dfs:
        for number, nodes_id in enumerate(branch):
            node = graph.nodes.get(nodes_id)["Node"]
            try:
                node_reward = reward_map[node]
            except KeyError:
                raise Exception(
                    "There is no node labeled : {label} in reward_map".format(label=node.label))
            mul *= node_reward

    return mul


class Reward:
    complex = get_graph_sum_complex_reward
    multiply = get_graph_mul_reward
