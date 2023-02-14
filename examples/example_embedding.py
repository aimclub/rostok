import networkx as nx

from tests.test_ruleset import get_terminal_graph_two_finger, get_terminal_graph_two_finger_mix, get_terminal_graph_three_finger
import app.rule_extention as re

get_terminal_graph_two_finger() == get_terminal_graph_two_finger_mix()

graph_1 = get_terminal_graph_two_finger()
graph_2 = get_terminal_graph_two_finger_mix()
graph_three = get_terminal_graph_three_finger()


vocab, __ = re.init_extension_rules()
node_vocab = vocab.node_vocab
name_nodes = node_vocab.node_dict.keys()
sorted_node_vocab = sorted(node_vocab.node_dict.keys())

dict_nodes = dict(enumerate(sorted_node_vocab))
None