""" Module contains NodeVocabulary class."""

from rostok.graph_grammar.node import Node, BlockWrapper, ROOT


class NodeVocabulary():
    """The class contains dictionary of nodes and methods to manipulate with it.
    
    This is a class to manage a dictionary of nodes. The keys are labels of the nodes and the values are Node objects
    User can create or add nodes to vocabulary, get individual nodes or list of nodes. 

    Attributes:
        node_dict (List[Node]): dictionary of all nodes.
        terminal_node_dict (List[Node]): dictionary of only terminal nodes. 
        non-terminal_node_dict (List[Node]): dictionary of only non-terminal nodes.     
    """

    def __init__(self):
        """Create an empty vocabulary."""
        self.node_dict = {}
        self.terminal_node_dict = {}
        self.nonterminal_node_dict = {}

    def add_node(self, node: Node):
        """Add an already created node to the vocabulary.

        Args:
            node (Node): node to be added to vocabulary.

        Raises:
            Exception: Attempt to add a Node with a label that is already in dictionary!
        """
        if node.label in self.node_dict.keys():
            raise Exception('Attempt to add a Node with a label that is already in dictionary!')

        self.node_dict[node.label] = node
        if node.is_terminal:
            self.terminal_node_dict[node.label] = node
        else:
            self.nonterminal_node_dict[node.label] = node

    def create_node(self,
                    label: str,
                    is_terminal: bool = False,
                    block_wrapper: BlockWrapper = None):
        """Create a node and add it to the vocabulary.
            
        Args:
            label (str): the label of the new node.
            is_terminal (bool, optional): defines if the new node is a terminal node. Default is False.
            block_wrapper (BlockWrapper, optional): the object that contains physical properties of the 
                node. Default is None. 

        Raises:
            Exception: Attempt to add a Node with a label that is already in dictionary!
        """

        if label in self.node_dict.keys():
            raise Exception('Attempt to create a Node with a label that is already in dictionary!')

        node = Node(label, is_terminal, block_wrapper)
        self.node_dict[label] = node
        if is_terminal:
            self.terminal_node_dict[label] = node
        else:
            self.nonterminal_node_dict[label] = node

    def get_node(self, label: str) -> Node:
        """Return a node corresponding to the label.

        Args:
            label(str): the label of the node that should be returned.

        Returns:
            A requested node as a Node class object.   
         
        Raises
            Exception: Node with given label not found!
        """

        node = self.node_dict.get(label, None)
        if node is None:
            raise Exception(f"Node with label {label} not found!")

        return node

    def check_node(self, label: str) -> bool:
        """Check if the label is in the vocabulary.

        Args:
            label(str): the label of the node that should be checked.
        
        Returns:
            bool: True is the label is in dictionary, False otherwise.
        """

        if label in self.node_dict.keys():
            return True
        else:
            return False

    def __str__(self):
        """Return the list of the labels in the vocabulary."""

        return str(self.node_dict.keys())

    def get_list_of_nodes(self, nodes: list[str]) -> list[Node]:
        """Returns list of Node objects corresponding to list of labels.
        
        Args:
            nodes (list[str]): list of labels to construct a list of Node objects.

        
        Returns:
            list of Node objects corresponding to the list of passed labels.
        """
        result = []
        for node in nodes:
            result.append(self.get_node(node))
        return result


if __name__ == '__main__':
    node_vocab = NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node('A')
    node_vocab.create_node('B')
    node_vocab.create_node('C')
    node_vocab.create_node('D')
    node_vocab.create_node('A1', is_terminal=True)
    node_vocab.create_node('B1', is_terminal=True)
    node_vocab.create_node('C1', is_terminal=True)
    print(node_vocab)
