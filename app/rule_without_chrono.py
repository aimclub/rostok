from rostok.graph_grammar.node import BlockWrapper, ROOT
from rostok.graph_grammar import node_vocabulary, rule_vocabulary


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np




def plot_graph(graph):
    plt.figure()
    nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(graph, dim=2), node_size=800,
                    labels={n: graph.nodes[n]["Node"].label for n in graph})
    #plt.figure()
    #nx.draw_networkx(graph, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800)
    plt.show()

def init_extension_rules():
    # %% Bodies for extansions rules
    width = [0.25, 0.35, 0.5]
    alpha = 45
    alpha_left = [0, 30, 60]
    alpha_right = [180, 150, 120]
    length_link = [0.4, 0.6, 0.8]


    # Nodes
    node_vocab = node_vocabulary.NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node("J")
    node_vocab.create_node("L")
    node_vocab.create_node("F")
    node_vocab.create_node("EM")
    node_vocab.create_node("SML")
    node_vocab.create_node("SMR")
    node_vocab.create_node("SMRP")
    node_vocab.create_node("SMRPA")
    node_vocab.create_node("SMLP")
    node_vocab.create_node("SMLPA")
    node_vocab.create_node("SMRM")
    node_vocab.create_node("SMRMA")
    node_vocab.create_node("SMLM")
    node_vocab.create_node("SMLMA")


    #O = Node("O")
    node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="L1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="L2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="L3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="F1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="F2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="F3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="U2", is_terminal=True, block_wrapper=None)

    node_vocab.create_node(label="TR1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TR2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TR3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRP1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRP2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRP3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRPA1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRPA2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRPA3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRM1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRM2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRM3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRMA1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRMA2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TRMA3", is_terminal=True, block_wrapper=None)

    node_vocab.create_node(label="TL1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TL2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TL3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLP1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLP2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLP3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLPA1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLPA2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLPA3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLM1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLM2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLM3", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLMA1", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLMA2", is_terminal=True, block_wrapper=None)
    node_vocab.create_node(label="TLMA3", is_terminal=True, block_wrapper=None)

    # Defines rules
    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

    rule_vocab.create_rule("InitMechanism_2", ["ROOT"], ["F", "SML", "SMR","EM","EM"], 0 , 0,[(0,1),(0,2),(1,3),(2,4)])
    rule_vocab.create_rule("InitMechanism_3_R", ["ROOT"], ["F", "SML", "SMRP","SMRM","EM","EM","EM"], 0 , 0,[(0,1),(0,2),(0,3),(1,4),(2,5),(3,6)])
    rule_vocab.create_rule("InitMechanism_3_R_A", ["ROOT"], ["F", "SML", "SMRPA","SMRMA","EM","EM","EM"], 0 , 0,[(0,1),(0,2),(0,3),(1,4),(2,5),(3,6)])
    rule_vocab.create_rule("InitMechanism_3_L", ["ROOT"], ["F", "SMLP","SMLM", "SMR","EM","EM","EM"], 0 , 0, [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6)])
    rule_vocab.create_rule("InitMechanism_3_L_A", ["ROOT"], ["F", "SMLPA","SMLMA", "SMR","EM","EM","EM"], 0 , 0, [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6)])
    rule_vocab.create_rule("InitMechanism_4", ["ROOT"], ["F", "SMLP","SMLM", "SMRP","SMRM","EM","EM","EM","EM"], 0 , 0, [(0,1),(0,2),(0,3),(0,4),(1,5),(2,6),(3,7),(4,8)])
    rule_vocab.create_rule("InitMechanism_4_A", ["ROOT"], ["F", "SMLPA","SMLMA", "SMRPA","SMRMA","EM","EM","EM","EM"], 0 , 0, [(0,1),(0,2),(0,3),(0,4),(1,5),(2,6),(3,7),(4,8)])
    rule_vocab.create_rule("FingerUpper", ["EM"], ["J", "L","EM"], 0 , 2, [(0,1),(1, 2)])

    rule_vocab.create_rule("TerminalFlat1", ["F"], ["F1"], 0 , 0)
    rule_vocab.create_rule("TerminalFlat2", ["F"], ["F2"], 0 , 0)
    rule_vocab.create_rule("TerminalFlat3", ["F"], ["F3"], 0 , 0)

    rule_vocab.create_rule("TerminalL1", ["L"], ["L1"], 0 , 0)
    rule_vocab.create_rule("TerminalL2", ["L"], ["L2"], 0 , 0)
    rule_vocab.create_rule("TerminalL3", ["L"], ["L3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformRight1", ["SMR"], ["TR1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRight2", ["SMR"], ["TR2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRight3", ["SMR"], ["TR3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformRightPlus1", ["SMRP"], ["TRP1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRightPlus2", ["SMRP"], ["TRP2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRightPlus3", ["SMRP"], ["TRP3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformRightPlusAngle1", ["SMRPA"], ["TRPA1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRightPlusAngle2", ["SMRPA"], ["TRPA2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRightPlusAngle3", ["SMRPA"], ["TRPA3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformRightMinus1", ["SMRM"], ["TRM1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRightMinus2", ["SMRM"], ["TRM2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRightMinus3", ["SMRM"], ["TRM3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformRightMinusAngle1", ["SMRMA"], ["TRMA1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRightMinusAngle2", ["SMRMA"], ["TRMA2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformRightMinusAngle3", ["SMRMA"], ["TRMA3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformLeft1", ["SML"], ["TL1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeft2", ["SML"], ["TL2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeft3", ["SML"], ["TL3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformLeftPlus1", ["SMLP"], ["TLP1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeftPlus2", ["SMLP"], ["TLP2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeftPlus3", ["SMLP"], ["TLP3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformLeftPlusAngle1", ["SMLPA"], ["TLPA1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeftPlusAngle2", ["SMLPA"], ["TLPA2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeftPlusAngle3", ["SMLPA"], ["TLPA3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformLeftMinus1", ["SMLM"], ["TLM1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeftMinus2", ["SMLM"], ["TLM2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeftMinus3", ["SMLM"], ["TLM3"], 0 , 0)

    rule_vocab.create_rule("TerminalTransformLeftMinusAngle1", ["SMLMA"], ["TLMA1"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeftMinusAngle2", ["SMLMA"], ["TLMA2"], 0 , 0)
    rule_vocab.create_rule("TerminalTransformLeftMinusAngle3", ["SMLMA"], ["TLMA3"], 0 , 0)

    rule_vocab.create_rule("TerminalEndLimb1", ["EM"], ["U1"], 0 , 0)
    rule_vocab.create_rule("TerminalEndLimb2", ["EM"], ["U2"], 0 , 0)
    rule_vocab.create_rule("TerminalJoint", ["J"], ["J1"], 0 , 0)

    return rule_vocab