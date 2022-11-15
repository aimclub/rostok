# imports from our code
import context
from engine import rule_vocabulary
from engine import node_vocabulary
from engine.node import ROOT, GraphGrammar, BlockWrapper
from engine.node_render import ChronoBody, ChronoTransform, ChronoRevolveJoint
from utils.transform_srtucture import FrameTransform

# imports from standard libs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# chrono imports
import pychrono as chrono
from pychrono import ChCoordsysD, ChVectorD, ChQuaternionD
from pychrono import Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X, \
    Q_ROTATE_Y_TO_X, Q_ROTATE_Y_TO_Z, \
    Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(graph, dim=2), node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.show()


# Define block types
mat = chrono.ChMaterialSurfaceNSC()
mat.SetFriction(0.5)
mat.SetDampingF(0.1)

# Bodies
link1 = BlockWrapper(ChronoBody, length=0.6)
link2 = BlockWrapper(ChronoBody, length=0.4)

flat1 = BlockWrapper(ChronoBody, width=0.8, length=0.2)
flat2 = BlockWrapper(ChronoBody, width=0.4, length=0.2)

u1 = BlockWrapper(ChronoBody, width=0.2, length=0.2)

# Transforms
RZX = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_X)
RZY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_Z_TO_Y)
RXY = ChCoordsysD(ChVectorD(0, 0, 0), Q_ROTATE_X_TO_Y)

MOVE_ZX_PLUS = FrameTransform([0.3, 0, 0.3], [1, 0, 0, 0])
MOVE_ZX_MINUS = FrameTransform([-0.3, 0, -0.3], [1, 0, 0, 0])

MOVE_X_PLUS = FrameTransform([0.3, 0, 0.], [1, 0, 0, 0])
MOVE_Z_PLUS_X_MINUS = FrameTransform([-0.3, 0, 0.3], [1, 0, 0, 0])

transform_rzx = BlockWrapper(ChronoTransform, RZX)
transform_rzy = BlockWrapper(ChronoTransform, RZY)
transform_rxy = BlockWrapper(ChronoTransform, RXY)
transform_mzx_plus = BlockWrapper(ChronoTransform, MOVE_ZX_PLUS)
transform_mzx_minus = BlockWrapper(ChronoTransform, MOVE_ZX_MINUS)
transform_mx_plus = BlockWrapper(ChronoTransform, MOVE_X_PLUS)
transform_mz_plus_x_minus = BlockWrapper(ChronoTransform, MOVE_Z_PLUS_X_MINUS)

type_of_input = ChronoRevolveJoint.InputType.Torque
# Joints
revolve1 = BlockWrapper(
    ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)


# %%
type_of_input = ChronoRevolveJoint.InputType.Torque

# Joints
revolve1 = BlockWrapper(
    ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z,  type_of_input)

app_node_vocab = node_vocabulary.NodeVocabulary()
app_node_vocab.add_node(ROOT)
app_node_vocab.create_node("J")
app_node_vocab.create_node("L")
app_node_vocab.create_node("F")
app_node_vocab.create_node("M")
app_node_vocab.create_node("EF")
app_node_vocab.create_node("EM")

app_node_vocab.create_node(
    label="J1", is_terminal=True, block_wrapper=revolve1)
app_node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=link1)
app_node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=link2)
app_node_vocab.create_node(label="F1", is_terminal=True, block_wrapper=flat1)
app_node_vocab.create_node(label="F2", is_terminal=True, block_wrapper=flat2)
app_node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=u1)
app_node_vocab.create_node(
    label="T1", is_terminal=True, block_wrapper=transform_mx_plus)
app_node_vocab.create_node(
    label="T2", is_terminal=True, block_wrapper=transform_mz_plus_x_minus)
app_node_vocab.create_node(
    label="T3", is_terminal=True, block_wrapper=transform_mzx_plus)
app_node_vocab.create_node(
    label="T4", is_terminal=True, block_wrapper=transform_mzx_minus)


app_rule_vocab = rule_vocabulary.RuleVocabulary(app_node_vocab)

app_rule_vocab.create_rule("FlatCreate", ["ROOT"], ["F"], 0, 0)
app_rule_vocab.create_rule(
    "Mount", ["F"], ["F", "M", "EM"], 0, 0, [(0, 1), (1, 2)])
app_rule_vocab.create_rule("MountAdd", ["M"], ["M", "EM"], 0, 1, [(0, 1)])
app_rule_vocab.create_rule("FingerUpper", ["EM"], [
                           "J", "L", "EM"], 0, 2, [(0, 1), (1, 2)])


app_rule_vocab.create_rule("TerminalFlat1", ["F"], ["F1"], 0, 0)
app_rule_vocab.create_rule("TerminalFlat2", ["F"], ["F2"], 0, 0)

app_rule_vocab.create_rule("TerminalL1", ["L"], ["L1"], 0, 0)
app_rule_vocab.create_rule("TerminalL2", ["L"], ["L2"], 0, 0)

app_rule_vocab.create_rule("TerminalTransformRX", ["M"], ["T1"], 0, 0)
app_rule_vocab.create_rule("TerminalTransformLZ", ["M"], ["T2"], 0, 0)
app_rule_vocab.create_rule("TerminalTransformR", ["M"], ["T3"], 0, 0)
app_rule_vocab.create_rule("TerminalTransformL", ["M"], ["T4"], 0, 0)
app_rule_vocab.create_rule("TerminalEndLimb", ["EM"], ["U1"], 0, 0)
app_rule_vocab.create_rule("TerminalJoint", ["J"], ["J1"], 0, 0)

 

# Required for criteria calc
list_J = list(map(app_node_vocab.get_node, ["J1"]))
list_RM = list(map(app_node_vocab.get_node, ["T1", "T3"]))
list_LM = list(map(app_node_vocab.get_node, ["T2", "T4"]))
list_B = list(map(app_node_vocab.get_node, ["L1", "L2", "F1", "F2", "U1"]))
node_features = [list_B, list_J, list_LM, list_RM]


def get_random_graph(n_iter: int, rule_vocab: rule_vocabulary.RuleVocabulary = app_rule_vocab, use_nonterminal_only: bool = True):
    G = GraphGrammar()
    for _ in range(n_iter//2+1):
        rules = rule_vocab.get_list_of_applicable_nonterminal_rules(G)
        if len(rules) > 0:
            rule = rule_vocab.get_rule(rules[np.random.choice(len(rules))])
            G.apply_rule(rule)
        else:
            break

    for _ in range(n_iter-(n_iter//2+1)):
        if use_nonterminal_only:
            rules = rule_vocab.get_list_of_applicable_nonterminal_rules(G)
        else:
            rules = rule_vocab.get_list_of_applicable_rules(G)
        if len(rules) > 0:
            rule = rule_vocab.get_rule(rules[np.random.choice(len(rules))])
            G.apply_rule(rule)
        else:
            break
    rule_vocab.make_graph_terminal(G)
    return G


def get_three_finger_graph():
    G = GraphGrammar()
    rule_action_non_terminal_three_finger = np.asarray(["FlatCreate", "Mount", "Mount", "Mount",
                                                        "FingerUpper", "FingerUpper", "FingerUpper",
                                                        "FingerUpper",  "FingerUpper", "FingerUpper"])
    rule_action_terminal_three_finger = np.asarray(["TerminalFlat1",
                                                    "TerminalL1", "TerminalL1", "TerminalL1",
                                                    "TerminalL2", "TerminalL2", "TerminalL2",
                                                    "TerminalTransformL", "TerminalTransformLZ",
                                                    "TerminalTransformRX",
                                                    "TerminalEndLimb", "TerminalEndLimb",
                                                    "TerminalEndLimb",
                                                    "TerminalJoint", "TerminalJoint", "TerminalJoint",
                                                    "TerminalJoint", "TerminalJoint", "TerminalJoint"])
    rule_action_three_finger = np.r_[
        rule_action_non_terminal_three_finger, rule_action_terminal_three_finger]
    rules = list(map(app_rule_vocab.get_rule, rule_action_three_finger))
    for i in list(rules):
        G.apply_rule(i)
    return G
