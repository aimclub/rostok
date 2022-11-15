# imports from our code
import context
from engine import rule_vocabulary
from engine import node_vocabulary
from engine.node import ROOT, GraphGrammar, BlockWrapper
import stubs.graph_environment as env
from engine.node_render import *

# imports from standard libs
import networkx as nx
import matplotlib.pyplot as plt
# chrono imports
import pychrono as chrono
from pychrono import ChCoordsysD, ChVectorD
from pychrono import (Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X,
                        Q_ROTATE_X_TO_Y)

import mcts


def plot_graph(graph:GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(graph, dim=2), node_size=800,
                    labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()

# Define block types
mat = chrono.ChMaterialSurfaceNSC()
mat.SetFriction(0.5)
mat.SetDampingF(0.1)

# Bodies
link1 = BlockWrapper(ChronoBody, length=0.3)
link2 = BlockWrapper(ChronoBody, length=0.2)

flat1 = BlockWrapper(ChronoBody, width=0.4, length=0.1)
flat2 = BlockWrapper(ChronoBody, width=0.7, length=0.1)

u1 = BlockWrapper(ChronoBody, width=0.1, length=0.1)

# Transforms
RZX = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_X)
RZY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_Y)
RXY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_X_TO_Y)

MOVE_ZX_PLUS = FrameTransform([0.3,0,0.3],[1,0,0,0])
MOVE_ZX_MINUS = FrameTransform([-0.3,0,-0.3],[1,0,0,0])

MOVE_X_PLUS = FrameTransform([0.3,0,0.],[1,0,0,0])
MOVE_Z_PLUS_X_MINUS = FrameTransform([-0.3,0,0.3],[1,0,0,0])

transform_rzx = BlockWrapper(ChronoTransform, RZX)
transform_rzy = BlockWrapper(ChronoTransform, RZY)
transform_rxy = BlockWrapper(ChronoTransform, RXY)
transform_mzx_plus = BlockWrapper(ChronoTransform, MOVE_ZX_PLUS)
transform_mzx_minus = BlockWrapper(ChronoTransform, MOVE_ZX_MINUS)
transform_mx_plus = BlockWrapper(ChronoTransform, MOVE_X_PLUS)
transform_mz_plus_x_minus = BlockWrapper(ChronoTransform, MOVE_Z_PLUS_X_MINUS)

type_of_input = ChronoRevolveJoint.InputType.Torque
# Joints
revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)


# %%
type_of_input = ChronoRevolveJoint.InputType.Position

# Joints
revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)

node_vocab = node_vocabulary.NodeVocabulary()
node_vocab.add_node(ROOT)
node_vocab.create_node("J")
node_vocab.create_node("L")
node_vocab.create_node("F")
node_vocab.create_node("M")
node_vocab.create_node("EF")
node_vocab.create_node("EM")

node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=revolve1)
node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=link1)
node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=link2)
node_vocab.create_node(label="F1", is_terminal=True, block_wrapper=flat1)
node_vocab.create_node(label="F2", is_terminal=True, block_wrapper=flat2)
node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=u1)
node_vocab.create_node(label="T1", is_terminal=True, block_wrapper=transform_mx_plus)
node_vocab.create_node(label="T2", is_terminal=True, block_wrapper=transform_mz_plus_x_minus)
node_vocab.create_node(label="T3", is_terminal=True, block_wrapper=transform_mzx_plus)
node_vocab.create_node(label="T4", is_terminal=True, block_wrapper=transform_mzx_minus)


rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

rule_vocab.create_rule("FlatCreate", ["ROOT"], ["F"], 0 , 0)
rule_vocab.create_rule("Mount", ["F"], ["F", "M", "EM"], 0 , 0, [(0,1), (1,2)])
rule_vocab.create_rule("MountAdd", ["M"], ["M", "EM"], 0 , 1, [(0,1)])
rule_vocab.create_rule("FingerUpper", ["EM"], ["J", "L", "EM"], 0 , 2, [(0,1), (1,2)])


rule_vocab.create_rule("TerminalFlat1", ["F"], ["F1"], 0 ,0)
rule_vocab.create_rule("TerminalFlat2", ["F"], ["F2"], 0 ,0)

rule_vocab.create_rule("TerminalL1", ["L"], ["L1"], 0 ,0)
rule_vocab.create_rule("TerminalL2", ["L"], ["L2"], 0 ,0)

rule_vocab.create_rule("TerminalTransformRX", ["M"], ["T1"], 0 ,0)
rule_vocab.create_rule("TerminalTransformLZ", ["M"], ["T2"], 0 ,0)
rule_vocab.create_rule("TerminalTransformR", ["M"], ["T3"], 0 ,0)
rule_vocab.create_rule("TerminalTransformL", ["M"], ["T4"], 0 ,0)
rule_vocab.create_rule("TerminalEndLimb", ["EM"], ["U1"], 0 ,0)
rule_vocab.create_rule("TerminalJoint", ["J"], ["J1"], 0 ,0)

rule_vocab.check_rules()

# %%
G = GraphGrammar()
max_numbers_rules = 5
# Create graph envirenments for algorithm (not gym)
graph_env = env.GraphVocabularyEnvironment(G, rule_vocab, max_numbers_rules)

# Hyperparameters: increasing: > error reward, < search time
time_limit = 1000
iteration_limit = 2000

# Initilize MCTS
searcher = mcts.mcts(timeLimit=time_limit)
finish = False

# Search until finding terminal mechanism with desired reward
while not finish:
    print("---")
    action = searcher.search(initialState=graph_env)
    finish, final_graph = graph_env.step(action)
    
# G = make_random_graph(20, rule_vocab, False)
print(final_graph.graph_partition_dfs())
plot_graph(final_graph)