"""Class RuleVocabulary."""

from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.graph_grammar.node import Rule, GraphGrammar, ROOT
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class RuleVocabulary():
    """The class that contains the rules for building the GraphGrammar object
    
    Attributes
    ----------
    node_vocab: NodeVocabulary
        the node vocabulary that should contain all the nodes used in the rules
    rule_dict: Dictionary{name: rule}
        the dictionary of all rules with the rule names as keys
    nonterminal_rule_dict: Dictionary{name: rule}
        the dictionary of nonterminal rules
    terminal_rule_dict: Dictionary{name: rule}
        the dictionary of terminal rules
    rules_nonterminal_node_set: Set(Node.label)
        the set of node labels that are used as the new nodes in the nonterminal rules. These are nodes that can appear in the final graph
    rules_terminal_node_set: Set(Node.label)
        the set of node labels that are used as the new nodes in the terminal rules. These are nodes that can appear in the final graph
    terminal_dict: dict{Node.label: [Node.label]}
        the dictionary of the terminal states of the nonterminal node
    completed: bool
        the state that changes in the check_rules() method if the set of rules is enough to make any graph terminal
    
    
    """
    def __init__(self, node_vocab: NodeVocabulary = NodeVocabulary()):
        """Cretae a new empty vocabulary object.
        
        Parameters
        ----------
        node_vocab: NodeVocabulary, optional
            the node vocabulary for the rules. Default - empty node vocabulary
        """
        self.node_vocab = node_vocab
        self.rule_dict = {}
        self.nonterminal_rule_dict = {}
        self.terminal_rule_dict = {}
        self.rules_nonterminal_node_set = set()
        self.rules_terminal_node_set = set()
        self.terminal_dict={}
        self.completed = False

    def create_rule(self, name:str, current_nodes: list[str], new_nodes: list[str], current_in_edges: int, current_out_edges: int, new_edges: list[(int,int)]=[], current_links: list[(int,int)]=[]):
        """Create a rule and add it to the dictionary.
        
        The method checks the created rule. There is no method to add already created rule to the vocabulary.

        Parameters
        ----------
        name:str
            name of the new rule
        current_nodes: list[str]
            list of the nodes to be replased
        new_nodes:
            list of the new nodes that to be inserted in the graph
        current_in_edges: int
            the node to link the in edges of the replaced node
        current_out_edges: int
            the node to link the out edges of the replaced node
        new_edges:list[(int, int)]
            the edges of the inserting subgraph

        Raises
        ------
        Exception
            If the name of the new rule is already in the vocabulary
        Exception
            If the length of the current_nodes is not 1
        Exception
            If the node vocabulary dose not include the replacing node or the new nodes
        Exception
            If the edge contains node idx out of graph length or the edge is a loop
        Exception
            Attempt to link in or out edges of the replaced node to the node idx out of new subgraph length
        """
        if name in self.rule_dict:
            raise Exception('This name is already in the rule vocabulary!')

        # Currently the GraphGrammar class can only apply rules that replace one node with the new system of nodes 
        # But in future we may apply replacement of the linked set of nodes
        if len(current_nodes)!=1:
            raise Exception(f'Prohibited length of the current_nodes: {len(current_nodes)}!')

        # Check that all nodes are in vocabulary     
        for node in current_nodes:
            if not self.node_vocab.check_node(node):
                raise Exception(f'Label {node} not in node vocabulary!')

        for node in new_nodes:
            if not self.node_vocab.check_node(node):
                raise Exception(f'Label {node} not in node vocabulary!')

        # if the rule deletes a node dont check its in and out connections
        if len(new_nodes)!=0:
             #Currently current_ins_links and current_out_links should be just numbers
            if current_in_edges > len(new_nodes)-1 or current_out_edges > len(new_nodes)-1 or current_in_edges < 0 or current_out_edges<0:
                raise Exception("Invalid linking of the in or out edges!")
            
        i=0
        # creating the new subgraph
        new_graph=nx.DiGraph()
        for label in new_nodes:
            new_graph.add_node(i, Node=self.node_vocab.get_node(label))
            i+=1
        
        for edge in new_edges:
            if edge[0] > len(new_nodes)-1 or edge[1]>len(new_nodes)-1 or edge[0] < 0 or edge[1] < 0 or edge[0] == edge[1]:
                raise Exception(f'Invalid edge {edge}')
            new_graph.add_edge(*edge)
            

        
        # graph_insert set the terminal status for the rule
        new_rule = Rule()
        new_rule.replaced_node = self.node_vocab.get_node(current_nodes[0])
        new_rule.graph_insert = new_graph
        new_rule.id_node_connect_parent = current_in_edges
        new_rule.id_node_connect_child = current_out_edges     
        self.rule_dict[name] = new_rule
        if new_rule._is_terminal:
            self.terminal_rule_dict[name] = new_rule
            self.rules_terminal_node_set.update(set(new_nodes))
            if len(new_nodes)>0:
                if current_nodes[0] in self.terminal_dict:
                    self.terminal_dict[current_nodes[0]].append(new_nodes[0])
                else:
                    self.terminal_dict[current_nodes[0]]=[new_nodes[0]]
        else:
            self.nonterminal_rule_dict[name] = new_rule
            #self.rules_nonterminal_node_set.update(set(current_nodes+new_nodes))
            self.rules_nonterminal_node_set.update(set(new_nodes))
    
    def __str__(self):
        """Print the rules dictionary of rules."""
        result=''
        for rule_tule in self.rule_dict.items():
            rule_graph = rule_tule[1].graph_insert
            rule_node = rule_tule[1].replaced_node
            result= result + rule_tule[0]+": "+rule_node.label+" ==> "+str([node[1]['Node'].label for node in rule_graph.nodes.items()]) +' '+ str([edge for edge in rule_graph.edges])+'\n'
        
        return result

    # Check set of rules itself, without any graph
    def check_rules(self):
        """Check set of rules itself, without any graph.
        
        Check the rules for having at least one terminal rule for every node that appears in the end graph of a nonterminal rule
        """
        # Check if all nonterminal nodes from vocab are in the rules. If not print a warning
        node_set=set(self.node_vocab.nonterminal_node_dict.keys())
        diff = node_set.difference(self.rules_nonterminal_node_set)
        if len(diff)>0:
            print(f"Nodes {diff} are not used as end nodes in the nonterminal rules!")
        
        # Check if all nodes in the end graphs of nonterminal rules have a terminal rule
        diff = self.rules_nonterminal_node_set.difference(set(self.terminal_dict.keys()))
        if len(diff)>0:
            print(f"Nodes {diff} don't have terminal rules! The set of rules is not completed!")
        else: self.completed = True

    def get_rule(self, name:str) -> Rule:
        """Return a rule with the corresponding name.
        
        Parameters
        ----------
        name: str
            the name of the rule to be returned
        """
        return self.rule_dict[name]

    def get_list_of_applicable_rules(self, grammar: GraphGrammar):
        """Return the total list of applicable rules for the current graph.

        Parameters
        ----------
        grammar: GraphGrammar
            A current GraphGrammar object 
        """
        list_of_applicable_rules = []
        for rule_tuple in self.rule_dict.items():
            rule = rule_tuple[1]
            rule_name = rule_tuple[0]
            label_to_replace = rule.replaced_node.label
            for node in grammar.nodes.items():
            #find a node that can be replaced using the rule
                if label_to_replace == node[1]['Node'].label:
                    list_of_applicable_rules.append(rule_name)
        
        return list_of_applicable_rules

    def get_list_of_applicable_nonterminal_rules(self, grammar:GraphGrammar):
        """Return the list of nonterminal applicable rules for the current graph.

        Parameters
        ----------
        grammar: GraphGrammar
            A current GraphGrammar object
        """
        list_of_applicable_rules = []
        for rule_tuple in self.nonterminal_rule_dict.items():
            rule = rule_tuple[1]
            rule_name = rule_tuple[0]
            label_to_replace = rule.replaced_node.label
            for node in grammar.nodes.items():
            #find a node that can be replaced using the rule
                if label_to_replace == node[1]['Node'].label:
                    list_of_applicable_rules.append(rule_name)
        
        return list_of_applicable_rules

    def get_list_of_applicable_terminal_rules(self, grammar:GraphGrammar):
        """Return the list of terminal applicable rules for the current graph.

        Parameters
        ----------
        grammar: GraphGrammar
            A current GraphGrammar object
        """
        list_of_applicable_rules = []
        for rule_tuple in self.terminal_rule_dict.items():
            rule = rule_tuple[1]
            rule_name = rule_tuple[0]
            label_to_replace = rule.replaced_node.label
            for node in grammar.nodes.items():
            #find a node that can be replaced using the rule
                if label_to_replace == node[1]['Node'].label:
                    list_of_applicable_rules.append(rule_name)
        
        return list_of_applicable_rules        
    
    def terminal_rules_for_node(self, node_name: str):
        """Return a list of the terminal rules for the node

        Parameters
        ----------
        node_name: str
            A node label for which function returns the list of the terminal rules 
        """
        rule_list = []
        for rule_name in self.terminal_rule_dict:
             if self.terminal_rule_dict[rule_name].replaced_node.label == node_name:
                rule_list.append(rule_name)
        
        return rule_list
    
    def make_graph_terminal(self, G:GraphGrammar):
        """Converts a graph into a graph with only terminal nodes with a random terminal rule for each node.
        
        Parameters
        ----------
        grammar: GraphGrammar
            A current GraphGrammar object
        """
        rule_list = []
        for node in G.nodes.items():
            if not node[1]["Node"].is_terminal:
                rules = self.terminal_rules_for_node(node[1]['Node'].label)
                rule=self.terminal_rule_dict[rules[np.random.choice(len(rules))]]
                rule_list.append(rule)
        for rule in rule_list:
            G.apply_rule(rule)


  
if __name__=='__main__':
    node_vocab = NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node('A')
    node_vocab.create_node('B')
    node_vocab.create_node('C')
    node_vocab.create_node('D')

    node_vocab.create_node('A1',is_terminal=True)
    node_vocab.create_node('B1', is_terminal=True)
    node_vocab.create_node('C1', is_terminal=True)
    rule_vocab = RuleVocabulary(node_vocab)
    rule_vocab.create_rule("First_Rule",['A'],['B','C'], 0, 1, [(0,1)])
    rule_vocab.create_rule("AT",['A'], ['A1'],0,0)
    rule_vocab.create_rule("BT",['B'], ['B1'],0,0)
    rule_vocab.create_rule("CT",['C'], ['C1'],0,0)
    rule_vocab.create_rule("ROOT",["ROOT"],["A"],0,0)
    rule_vocab.create_rule("CD",["C"], [], 0, 0)
    print(rule_vocab)

    rule_vocab.check_rules()
    G = GraphGrammar()
    G.apply_rule(rule_vocab.get_rule("ROOT"))
    G.apply_rule(rule_vocab.get_rule("First_Rule"))
    G.apply_rule(rule_vocab.get_rule("CD"))
    rule_vocab.make_graph_terminal(G)

    plt.figure()
    nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G, dim=2), node_size=800,
                    labels={n: G.nodes[n]["Node"].label for n in G})

    plt.show()
    


         