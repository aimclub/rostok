"""Class NodeVocabulary"""
import context 
from engine.node import Node, BlockWrapper, ROOT


class NodeVocabulary():
    """A class to contain nodes for robot graphs.
    
    Attributes
    ----------
    node_dict: List[Node]
        dictionary of all nodes 
    terminal_node_dict:  List[Node]
        dictionary of terminal nodes 
    nonterminal_node_dict:  List[Node]
        dictionary of nonterminal nodes 

    Methods
    -------
    add_node(node: Node)
        Adds a node that already created
    create_node(self, label: str, is_terminal:bool=False, block_wrapper:BlockWrapper=None):
        Create a node and add it to the vocabulary.
    def get_node(self, label:str) -> Node:
        Return a node corresponding to the label.
    check_node(self, label:str)->bool:
        Check if the label is in the vocabulary.
    
    """
 
    def __init__(self):
        """Create an empty vocabulary. Currently no parameters."""
        self.node_dict = {}
        self.terminal_node_dict = {}
        self.nonterminal_node_dict = {}

    def add_node(self, node: Node):
        """Add an already created node to the vocabulary.

        Parameters
        ----------
        node: Node
            node to be added to vocabulary
        
        Raises
        ------
        Exception
            If the label of the new node is already in the vocabulary 
        """
        if node.label in self.node_dict.keys():
            raise Exception('Attempt to add a Node with a label that is already in dictionary!')

        self.node_dict[node.label] = node
        if node.is_terminal: self.terminal_node_dict[node.label] = node
        else:  self.nonterminal_node_dict[node.label] = node

    def create_node(self, label: str, is_terminal:bool=False, block_wrapper:BlockWrapper=None):
        """Create a node and add it to the vocabulary.

        Parameters
        ----------
        label: str
            the label of the new node
        is_terminal:bool, optional
            defines if the new node is a terminal node. Default is False
        block_wrapper:
            the wrapper of the node that would be used to create a chrono object for simulation

        Raises
        ------
        Exception
            If the label of the new node is already in the vocabulary 
        """
        if label in self.node_dict.keys():
            raise Exception('Node with this label already exists!')

        node = Node(label, is_terminal, block_wrapper)         
        self.node_dict[label] = node
        if is_terminal: self.terminal_node_dict[label] = node
        else:  self.nonterminal_node_dict[label] = node

    def get_node(self, label:str) -> Node:
        """Return a node corresponding to the label.

        Parameters
        ----------
        label:str
            the label of the node that should be returned

        Raises
        ------
        Exception
            If the node with label doesn't exist in the vocabulary
        """
        node = self.node_dict.get(label, None)
        if node is None:
            raise Exception(f"Node with label {label} not found!")

        return node

    def check_node(self, label:str)->bool:
        """Check if the label is in the vocabulary.

        Parameters
        ----------
        label:str
            the label of the node that should be checked
        """
        if label in self.node_dict.keys(): return True
        else: return False

    def __str__(self):
        """Return the list of the labels in the vocabulary."""
        return str(self.node_dict.keys())


if __name__ == '__main__':
    node_vocab = NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node('A')
    node_vocab.create_node('B')
    node_vocab.create_node('C')
    node_vocab.create_node('D')
    node_vocab.create_node('A1',is_terminal=True)
    node_vocab.create_node('B1', is_terminal=True)
    node_vocab.create_node('C1', is_terminal=True)
    print(node_vocab)


    