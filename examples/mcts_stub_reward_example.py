import context
import stubs.graph_environment as env_graph
import mcts
import networkx as nx
import matplotlib.pyplot as plt

from engine.node import Node, Rule, GraphGrammar, ROOT
from stubs.graph_reward import Reward


J = Node("J")
L = Node("L")
P = Node("P")
U = Node("U")
M = Node("M")
EF = Node("EF")
EM = Node("EM")

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

# Terminal nodes

J1 = Node("J1", is_terminal=True)
M1 = Node("M1", is_terminal=True)
U1 = Node("U1", is_terminal=True)
L1 = Node("L1", is_terminal=True)
P1 = Node("P1", is_terminal=True)
EM1 = Node("EM1", is_terminal=True)
EF1 = Node("EF1", is_terminal=True)

TerminalJ1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=J1)
TerminalJ1.id_node_connect_child = 0
TerminalJ1.id_node_connect_parent = 0
TerminalJ1.graph_insert = rule_graph
TerminalJ1.replaced_node = J

TerminalL1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=L1)
TerminalL1.id_node_connect_child = 0
TerminalL1.id_node_connect_parent = 0
TerminalL1.graph_insert = rule_graph
TerminalL1.replaced_node = L

TerminalU1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=U1)
TerminalU1.id_node_connect_child = 0
TerminalU1.id_node_connect_parent = 0
TerminalU1.graph_insert = rule_graph
TerminalU1.replaced_node = U

TerminalM1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=M1)
TerminalM1.id_node_connect_child = 0
TerminalM1.id_node_connect_parent = 0
TerminalM1.graph_insert = rule_graph
TerminalM1.replaced_node = M

TerminalP1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=P1)
TerminalP1.id_node_connect_child = 0
TerminalP1.id_node_connect_parent = 0
TerminalP1.graph_insert = rule_graph
TerminalP1.replaced_node = P

TerminalEM1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=EM1)
TerminalEM1.id_node_connect_child = 0
TerminalEM1.id_node_connect_parent = 0
TerminalEM1.graph_insert = rule_graph
TerminalEM1.replaced_node = EM

TerminalEF1 = Rule()
rule_graph = nx.DiGraph()
rule_graph.add_node(0, Node=EF1)
TerminalEF1.id_node_connect_child = 0
TerminalEF1.id_node_connect_parent = 0
TerminalEF1.graph_insert = rule_graph
TerminalEF1.replaced_node = EF

DeliteEndMount = Rule()
rule_graph = nx.DiGraph()
DeliteEndMount.id_node_connect_child = 0
DeliteEndMount.id_node_connect_parent = 0
DeliteEndMount.graph_insert = rule_graph
DeliteEndMount.replaced_node = EM

G = GraphGrammar()
rule_action = [PalmCreate, Mount, MountAdd, MountUpper, FingerUpper, DeliteEndMount,  # Non terminal
               TerminalJ1, TerminalL1, TerminalM1, TerminalP1, TerminalU1, TerminalEM1, TerminalEF1]  # Terminal
max_numbers_rules = 10

# Create graph envirenments for algorithm (not gym)
env = env_graph.GraphEnvironment(G, rule_action, max_numbers_rules)

# Hyperparameters: increasing: > error reward, < search time
time_limit = 1000
iteration_limit = 2000

# Initilize MCTS
searcher = mcts.mcts(timeLimit=time_limit)
finish = False


reward_map_2 = {J1: 1, L1: 2, P1: 1, U1: 1, M1: 4, EF1: -2, EM1: -5}
env.set_node_rewards(reward_map_2, Reward.complex)

# Search until finding terminal mechanism with desired reward
while not finish:
    print("---")
    action = searcher.search(initialState=env)
    finish, final_graph = env.step(action)

# Plot final graph
plt.figure()
nx.draw_networkx(final_graph, pos=nx.kamada_kawai_layout(final_graph, dim=2), node_size=800,
                 labels={n: final_graph.nodes[n]["Node"].label for n in final_graph})

plt.show()
